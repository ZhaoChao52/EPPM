import math

import torch
from torch import nn

from backbone import ACPTGWUpdate, MGPTUnifiedModel


class ParallelExpertMGPT(MGPTUnifiedModel):


    def __init__(self, config, num_items, num_behaviors, result_dir):
        base_config = dict(config, model_variant="base")
        super().__init__(base_config, num_items, num_behaviors, result_dir)
        self.config_dict = config
        self.acpt_update = ACPTGWUpdate(
            config["hidden_size"],
            num_behaviors,
            config["transition_rank"],
            config["gate_hidden_size"],
        )
        self.expert_mix_max = float(config.get("expert_mix_max", 0.25))
        initial_mix = float(config.get("expert_mix_initial", 0.02))
        if not 0.0 < initial_mix < self.expert_mix_max:
            raise ValueError("expert_mix_initial must be in (0, expert_mix_max)")
        initial_probability = initial_mix / self.expert_mix_max
        self.expert_mix_logit = nn.Parameter(
            torch.tensor(math.log(initial_probability / (1.0 - initial_probability)))
        )
        self.register_buffer("completed_updates", torch.zeros((), dtype=torch.long))
        self.warmup_steps = int(config["gate_warmup_steps"])
        self.last_diagnostics = {}

    @property
    def expert_mix(self):
        return self.expert_mix_max * torch.sigmoid(self.expert_mix_logit)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.completed_updates.fill_(self.global_step)

    def _logits(self, batch):

        joint_states = self.backbone(batch["prefix_ids"], batch["prefix_behaviors"])


        pre_states = self.backbone(batch["pre_ids"], batch["pre_behaviors"])
        anchor_item = batch["anchor_item"]
        anchor_behavior = batch["anchor_behavior"]
        item_embedding = self.backbone.item_embedding(anchor_item)
        behavior_embedding = self.backbone.type_embedding(anchor_behavior)
        warmup = int(self.completed_updates) < self.warmup_steps

        acpt_candidate, gate, transition = self.acpt_update(
            pre_states[-1],
            item_embedding,
            behavior_embedding,
            anchor_behavior,
            warmup,
        )
        mix = self.expert_mix
        expert_state = joint_states[-1] + mix * (acpt_candidate - joint_states[-1])

        item_weights = self.backbone.item_embedding.weight
        base_level_logits = torch.stack(
            [torch.matmul(state, item_weights.t()) for state in joint_states], dim=1
        )
        base_logits = base_level_logits.max(dim=1).values
        expert_logits = torch.matmul(expert_state, item_weights.t())
        logits = torch.maximum(base_logits, expert_logits)
        logits[:, 0] = torch.finfo(logits.dtype).min

        with torch.no_grad():
            self.last_diagnostics = {
                "gate_mean": float(gate.mean()),
                "expert_mix": float(mix),
                "transition_norm": float(transition.norm(dim=-1).mean()),
                "candidate_delta_norm": float(
                    (acpt_candidate - joint_states[-1]).norm(dim=-1).mean()
                ),
                "effective_expert_delta_norm": float(
                    (expert_state - joint_states[-1]).norm(dim=-1).mean()
                ),
                "expert_win_rate": float((expert_logits > base_logits).float().mean()),
            }
        return logits

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        if batch_idx % 50 == 0:
            values = " ".join(
                f"{key}={value:.6g}" for key, value in self.last_diagnostics.items()
            )
            print(
                f"[PARALLEL_EXPERT] epoch={self.current_epoch} "
                f"step={self.global_step} loss={float(loss.detach()):.6g} {values}",
                flush=True,
            )
        return loss

    def on_test_start(self):
        print(
            "[TEST_GATE] completed_updates=%d warmup=%s expert_mix=%.6g"
            % (
                int(self.completed_updates),
                int(self.completed_updates) < self.warmup_steps,
                float(self.expert_mix),
            ),
            flush=True,
        )
