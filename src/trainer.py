from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def build_trainer(config):
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints",
        monitor="Val:NDCG@20",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="best",
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )
    early_stop = EarlyStopping(
        monitor="Val:NDCG@20",
        mode="max",
        patience=int(config["patience"]),
        min_delta=0.0,
        check_on_train_epoch_end=False,
    )
    return pl.Trainer(
        default_root_dir=str(output),
        accelerator="gpu",
        devices=1,
        max_epochs=int(config["max_epochs"]),
        check_val_every_n_epoch=int(config["check_val_every_n_epoch"]),
        callbacks=[checkpoint, early_stop],
        logger=False,
        enable_progress_bar=False,
        log_every_n_steps=50,
        precision=32,
        accumulate_grad_batches=int(config.get("accumulate_grad_batches", 1)),
    )
