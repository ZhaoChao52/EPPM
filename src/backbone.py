import copy
import csv
import json
import math
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.bias, std=0.02)

    def forward(self, inputs, graph):
        return graph.matmul(inputs.matmul(self.weight) + self.bias)


class MLGCN(nn.Module):
    def __init__(self, hidden_size, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.gcn = GCN(hidden_size)

    def forward(self, inputs, graph, levels):
        outputs = []
        state = F.dropout(inputs, self.dropout, training=self.training)
        outputs.append(state)
        for _ in range(levels - 1):
            state = F.leaky_relu(self.gcn(state, graph))
            state = F.dropout(state, self.dropout, training=self.training)
            outputs.append(state)
        return outputs


class FeedForward(nn.Module):
    def __init__(self, hidden_size, inner_size, dropout, activation, layer_norm_eps):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.activation = {
            "gelu": F.gelu,
            "relu": F.relu,
            "swish": lambda value: value * torch.sigmoid(value),
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }[activation]

    def forward(self, inputs):
        state = self.dense_2(self.activation(self.dense_1(inputs)))
        return self.layer_norm(self.dropout(state) + inputs)


class LinMHA(nn.Module):
    def __init__(self, n_heads, hidden_size, hidden_dropout, attn_dropout, eps):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.num_heads = n_heads
        self.head_size = hidden_size // n_heads
        self.sqrt_head_size = math.sqrt(self.head_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.dropout = nn.Dropout(hidden_dropout)

    def _heads(self, values):
        return values.view(*values.size()[:-1], self.num_heads, self.head_size)

    def forward(self, inputs, attention_mask):
        normalized = self.dropout(self.layer_norm(inputs))
        query = self._heads(self.query(normalized)).permute(0, 2, 1, 3)
        key = self._heads(self.key(normalized)).permute(0, 2, 3, 1)
        value = self._heads(self.value(normalized)).permute(0, 2, 1, 3)
        query = F.elu(query)
        key = F.elu(key)
        query = query / (torch.norm(query, dim=3, keepdim=True) + 1e-12)
        key = key / (torch.norm(key, dim=2, keepdim=True) + 1e-12)
        context = torch.matmul(query, torch.matmul(key, value)) / self.sqrt_head_size
        context = context.permute(0, 2, 1, 3).contiguous().view_as(inputs)
        state = self.dropout(self.dense(context))
        return self.layer_norm(state + normalized)


class MultiQueryMHA(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout,
        attn_dropout,
        eps,
        user_level,
        l_p,
        agg,
    ):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.num_attention_heads = n_heads
        self.attention_head_size = hidden_size // n_heads
        self.all_head_size = hidden_size
        self.user_level = user_level
        self.l_p = l_p
        self.agg = agg
        self.multi_query = nn.ModuleList()
        for level in range(user_level):
            width = (level + 1) * hidden_size if agg == "con" else hidden_size
            self.multi_query.append(nn.Linear(width, hidden_size))
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)
        self.out_dropout = nn.Dropout(hidden_dropout)


class DeviceSafeMultiQueryMHA(MultiQueryMHA):


    def forward(self, input_tensor, scaled_num, mask):
        mask = mask.squeeze(1)
        device = input_tensor.device
        batch_index = torch.arange(input_tensor.size(0), device=device)
        scaled_num = scaled_num.to(device)
        max_index = input_tensor.size(1) - 1
        hts = []
        for i in range(self.user_level):
            selected = [
                input_tensor[
                    batch_index,
                    torch.clamp(scaled_num - (j + 1), -1, max_index),
                ]
                for j in range(i + 1)
            ]
            if self.agg == "con":
                query_input = torch.cat(selected, dim=-1)
            else:
                query_input = torch.stack(selected, dim=1).sum(dim=1)
            hts.append(self.multi_query[i](query_input).unsqueeze(1))

        multi_query_input = torch.cat(hts, dim=1)
        multi_query_layer = multi_query_input.div(
            torch.norm(multi_query_input, p=2, dim=1, keepdim=True) + 1e-12
        )
        mixed_query = self.query(multi_query_layer)
        mixed_key = self.key(input_tensor)
        mixed_value = self.value(input_tensor)
        query = mixed_query.view(-1, mixed_query.size(1), self.attention_head_size)
        key = mixed_key.view(-1, mixed_key.size(1), self.attention_head_size)
        value = mixed_value.view(-1, mixed_value.size(1), self.attention_head_size)
        attention_map = torch.sigmoid(torch.matmul(query, key.permute(0, 2, 1)))
        attention_map = attention_map.view(
            -1, query.size(1) * self.num_attention_heads, input_tensor.size(1)
        ).permute(0, 2, 1)
        attention_map = torch.softmax(2 * attention_map, dim=1)
        attention_diagnostic = attention_map.view(
            -1, self.num_attention_heads, input_tensor.size(1), self.user_level
        )
        pool = nn.LPPool1d(self.l_p, self.user_level, stride=self.user_level)
        attention_map = pool(attention_map)
        attention_map = torch.masked_fill(
            attention_map, ~mask.bool().unsqueeze(-1), float("-inf")
        )
        attention_map = torch.softmax(2 * attention_map, dim=1)
        attention_map = torch.nan_to_num(attention_map)
        value = value.view(
            input_tensor.size(0),
            input_tensor.size(1),
            self.num_attention_heads,
            self.attention_head_size,
        )
        output = (attention_map.unsqueeze(-1) * value).reshape(
            input_tensor.size(0), input_tensor.size(1), self.all_head_size
        )
        output = output * mask.view(mask.size(0), -1, 1).float()
        output = self.dense(output)
        output = self.out_dropout(output)
        output = self.LayerNorm(output + input_tensor)
        return output, attention_diagnostic


class LengthAwareMixer(nn.Module):


    def __init__(
        self,
        seq_len,
        scales,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
        user_level,
        l_p,
        agg,
    ):
        super().__init__()
        assert seq_len % scales[0] == 0 and seq_len % scales[1] == 0
        self.seq_len = int(seq_len)
        self.scales = list(scales)
        self.out_fc = nn.Linear(3 * seq_len, seq_len)
        self.attention1 = LinMHA(
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
        )
        self.attention2 = DeviceSafeMultiQueryMHA(
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
            user_level[0],
            l_p,
            agg,
        )
        self.attention3 = DeviceSafeMultiQueryMHA(
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
            user_level[1],
            l_p,
            agg,
        )

    @staticmethod
    def _segment_lengths(scale, seq_len, mask):
        counts = mask.squeeze(1).sum(dim=1).long()
        width = seq_len // scale
        segment_lengths = []
        for segment in range(scale):
            segment_lengths.append((counts - segment * width).clamp(0, width))
        return torch.stack(segment_lengths, dim=1)

    def _scale_path(self, inputs, positions, mask, attention, scale):
        width = self.seq_len // scale
        lengths = self._segment_lengths(scale, self.seq_len, mask)
        outputs = []
        for segment in range(scale):
            start = segment * width
            end = start + width
            output, _ = attention(
                inputs[:, start:end], lengths[:, segment], mask[:, :, start:end]
            )
            outputs.append(output + positions[:, start:end])
        return torch.cat(outputs, dim=1)

    def forward(self, inputs, positions, mask):
        fine = self.attention1(inputs + positions, mask)
        medium = self._scale_path(
            inputs, positions, mask, self.attention2, self.scales[0]
        )
        coarse = self._scale_path(
            inputs, positions, mask, self.attention3, self.scales[1]
        )
        output = torch.cat([fine, medium, coarse], dim=1).transpose(1, 2)
        return self.out_fc(output).transpose(1, 2)


class UnifiedMGPTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = LengthAwareMixer(
            config["model_sequence_length"],
            config["scales"],
            config["n_heads"],
            config["hidden_size"],
            config["hidden_dropout_prob"],
            config["attn_dropout_prob"],
            config["layer_norm_eps"],
            config["user_level"],
            config["l_p"],
            config["agg"],
        )
        self.feed_forward = FeedForward(
            config["hidden_size"],
            config["inner_size"],
            config["hidden_dropout_prob"],
            config["hidden_act"],
            config["layer_norm_eps"],
        )

    def forward(self, states, positions, mask):
        return self.feed_forward(self.attention(states, positions, mask))


class UnifiedMGPTBackbone(nn.Module):
    def __init__(self, config, num_items, num_behaviors):
        super().__init__()
        self.config = config
        self.num_items = int(num_items)
        self.model_len = int(config["model_sequence_length"])
        self.item_level = int(config["item_level"])
        hidden = int(config["hidden_size"])
        self.item_embedding = nn.Embedding(self.num_items + 1, hidden, padding_idx=0)
        self.type_embedding = nn.Embedding(num_behaviors + 1, hidden, padding_idx=0)
        self.position_embedding = nn.Embedding(self.model_len, hidden)
        self.graph = MLGCN(hidden)
        layer = UnifiedMGPTLayer(config)
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(int(config["n_layers"]))]
        )
        self.initializer_range = float(config["initializer_range"])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _right_pad(self, values):
        batch, source_len = values.shape
        lengths = values.ne(0).sum(dim=1)
        positions = torch.arange(self.model_len, device=values.device).unsqueeze(0)
        starts = source_len - lengths
        source_positions = (starts.unsqueeze(1) + positions).clamp(0, source_len - 1)
        gathered = values.gather(1, source_positions)
        return gathered.masked_fill(positions >= lengths.unsqueeze(1), 0), lengths

    def forward(self, item_ids, behavior_ids):
        item_ids, lengths = self._right_pad(item_ids)
        behavior_ids, behavior_lengths = self._right_pad(behavior_ids)
        assert torch.equal(lengths, behavior_lengths)
        mask = item_ids.ne(0)
        item = self.item_embedding(item_ids)
        behavior = self.type_embedding(behavior_ids)


        interaction = item * behavior
        adjacency = torch.sigmoid(torch.matmul(interaction, interaction.transpose(1, 2)))
        adjacency = adjacency * mask.unsqueeze(1) * mask.unsqueeze(2)
        adjacency = adjacency + torch.eye(
            self.model_len, device=item_ids.device
        ).unsqueeze(0)
        row_sum = adjacency.sum(dim=1)
        inv_sqrt = torch.where(
            row_sum > 0, row_sum.rsqrt(), torch.zeros_like(row_sum)
        )
        norm_adjacency = torch.diag_embed(inv_sqrt).matmul(adjacency)
        graph_levels = self.graph(item + behavior, norm_adjacency, self.item_level)

        positions = self.position_embedding(
            torch.arange(self.model_len, device=item_ids.device)
        ).unsqueeze(0).expand(item_ids.size(0), -1, -1)
        attention_mask = mask.long().unsqueeze(1)
        outputs = []
        batch_index = torch.arange(item_ids.size(0), device=item_ids.device)
        last_index = (lengths - 1).clamp_min(0)
        for graph_state in graph_levels:
            state = graph_state
            for layer in self.layers:
                state = layer(state, positions, attention_mask)
            outputs.append(state[batch_index, last_index])
        return outputs

    def logits(self, item_ids, behavior_ids):
        states = self(item_ids, behavior_ids)
        return self.logits_from_states(states)

    def logits_from_states(self, states):
        item_weights = self.item_embedding.weight
        logits = torch.stack(
            [torch.matmul(state, item_weights.t()) for state in states], dim=1
        ).max(dim=1).values
        logits[:, 0] = torch.finfo(logits.dtype).min
        return logits


class ACPTGWUpdate(nn.Module):
    def __init__(self, hidden_size, num_behaviors, rank, gate_hidden_size):
        super().__init__()
        self.rank = int(rank)
        self.transition_u = nn.Parameter(
            torch.empty(num_behaviors + 1, hidden_size, self.rank)
        )
        self.transition_v = nn.Parameter(
            torch.empty(num_behaviors + 1, hidden_size, self.rank)
        )
        self.item_projection = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Sequential(
            nn.Linear(3 * hidden_size, int(gate_hidden_size)),
            nn.GELU(),
            nn.Linear(int(gate_hidden_size), 1),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        nn.init.normal_(self.transition_u, std=0.02)
        nn.init.normal_(self.transition_v, std=0.02)

    def transition(self, state, item_embedding, behavior_embedding, behavior, warmup):
        transition_u = self.transition_u[behavior]
        transition_v = self.transition_v[behavior]
        low_rank = torch.einsum("bhr,bh->br", transition_v, state)
        transition = torch.einsum("bhr,br->bh", transition_u, low_rank)
        learned_gate = torch.sigmoid(
            self.gate(torch.cat([state, item_embedding, behavior_embedding], dim=-1))
        )
        gate = torch.ones_like(learned_gate) if warmup else learned_gate
        return gate, transition

    def forward(self, state, item_embedding, behavior_embedding, behavior, warmup):
        gate, transition = self.transition(
            state, item_embedding, behavior_embedding, behavior, warmup
        )
        updated = self.layer_norm(
            state + self.item_projection(item_embedding) + gate * transition
        )
        return updated, gate, transition


class MGPTUnifiedModel(pl.LightningModule):
    def __init__(self, config, num_items, num_behaviors, result_dir):
        super().__init__()
        self.config_dict = config
        self.backbone = UnifiedMGPTBackbone(config, num_items, num_behaviors)
        self.model_variant = config.get("model_variant", "base")
        if self.model_variant not in ("base", "acpt_gw", "dual_state_acpt_gw"):
            raise ValueError(f"Unknown model_variant: {self.model_variant}")
        self.gate_warmup_epochs = int(config.get("gate_warmup_epochs", 0))
        self.acpt_update = None
        if self.model_variant in ("acpt_gw", "dual_state_acpt_gw"):
            self.acpt_update = ACPTGWUpdate(
                int(config["hidden_size"]),
                num_behaviors,
                int(config["transition_rank"]),
                int(config.get("gate_hidden_size", config["hidden_size"])),
            )
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.loss_fn = nn.CrossEntropyLoss()
        self._outputs = []
        self._epoch_started = None

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=float(self.config_dict["learning_rate"]),
            weight_decay=float(self.config_dict["weight_decay"]),
        )

    def on_train_epoch_start(self):
        self._epoch_started = time.perf_counter()
        sampler = self.trainer.train_dataloader.sampler
        sampler.set_epoch(self.current_epoch)
        print("[SAMPLER] " + json.dumps(sampler.audit(), sort_keys=True), flush=True)

    def training_step(self, batch, batch_idx):
        logits = self._logits(batch)
        loss = self.loss_fn(logits, batch["target"])
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        elapsed = time.perf_counter() - self._epoch_started
        print(
            "[EPOCH] "
            + json.dumps(
                {
                    "epoch": int(self.current_epoch),
                    "global_step": int(self.global_step),
                    "seconds": elapsed,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated()
                    if self.device.type == "cuda"
                    else 0,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def on_validation_epoch_start(self):
        self._outputs = []

    def validation_step(self, batch, batch_idx):
        self._evaluation_step(batch)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self._outputs = []
            return
        self._finish_evaluation("Val")

    def on_test_epoch_start(self):
        self._outputs = []

    def test_step(self, batch, batch_idx):
        self._evaluation_step(batch)

    def on_test_epoch_end(self):
        self._finish_evaluation("Test")

    def _evaluation_step(self, batch):
        logits = self._logits(batch)
        target = batch["target"]
        topk = torch.topk(logits[:, 1:], k=20, dim=1).indices + 1
        matches = topk.eq(target.unsqueeze(1))
        hit20 = matches.any(dim=1).float()
        ranks = matches.float().argmax(dim=1)
        hit10 = (matches[:, :10].any(dim=1)).float()
        ndcg20 = hit20 / torch.log2(ranks.float() + 2.0)
        ndcg10 = hit10 / torch.log2(ranks.float() + 2.0)
        self._outputs.append(
            {
                "recall10": hit10.cpu(),
                "recall20": hit20.cpu(),
                "ndcg10": ndcg10.cpu(),
                "ndcg20": ndcg20.cpu(),
            }
        )

    def _logits(self, batch):
        if self.model_variant == "base":
            return self.backbone.logits(
                batch["prefix_ids"], batch["prefix_behaviors"]
            )
        states = self.backbone(batch["pre_ids"], batch["pre_behaviors"])
        anchor_item = batch["anchor_item"]
        anchor_behavior = batch["anchor_behavior"]
        item_embedding = self.backbone.item_embedding(anchor_item)
        behavior_embedding = self.backbone.type_embedding(anchor_behavior)
        warmup = self.current_epoch < self.gate_warmup_epochs
        if self.model_variant == "dual_state_acpt_gw":
            joint_states = self.backbone(
                batch["prefix_ids"], batch["prefix_behaviors"]
            )
            updated_states = []
            for pre_state, joint_state in zip(states, joint_states):
                gate, transition = self.acpt_update.transition(
                    pre_state,
                    item_embedding,
                    behavior_embedding,
                    anchor_behavior,
                    warmup,
                )
                updated_states.append(
                    self.acpt_update.layer_norm(joint_state + gate * transition)
                )
            return self.backbone.logits_from_states(updated_states)
        updated_states = []
        for state in states:
            updated, _, _ = self.acpt_update(
                state,
                item_embedding,
                behavior_embedding,
                anchor_behavior,
                warmup,
            )
            updated_states.append(updated)
        return self.backbone.logits_from_states(updated_states)

    def _finish_evaluation(self, prefix):
        merged = {
            key: torch.cat([output[key] for output in self._outputs])
            for key in self._outputs[0]
        }
        metrics = {
            f"{prefix}:Recall@10": float(merged["recall10"].mean()),
            f"{prefix}:NDCG@10": float(merged["ndcg10"].mean()),
            f"{prefix}:Recall@20": float(merged["recall20"].mean()),
            f"{prefix}:NDCG@20": float(merged["ndcg20"].mean()),
            "epoch": int(self.current_epoch),
            "params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        for key, value in metrics.items():
            if key.startswith(prefix):
                self.log(key, value, on_epoch=True, sync_dist=True)
        print(f"[{prefix.upper()}] " + json.dumps(metrics, sort_keys=True), flush=True)
        output_path = self.result_dir / (
            "test_metrics.json" if prefix == "Test" else "last_val_metrics.json"
        )
        output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        metrics_path = self.result_dir / "metrics.csv"
        exists = metrics_path.exists()
        with metrics_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics))
            if not exists:
                writer.writeheader()
            writer.writerow(metrics)
        self._outputs = []
