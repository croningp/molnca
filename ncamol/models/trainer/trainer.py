import os
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torch.nn as nn

from ncamol import SEED
from ..model import LitModelExtForce, LitModel


def lightning_train_loop(
    x0: torch.Tensor,
    target: torch.Tensor,
    model: nn.Module | None = None,
    learning_rate: float = 1e-4,
    normal_std: float = 0.01,
    alive_threshold: float = 0.1,
    cell_fire_rate: float = 0.5,
    num_epochs: int = 100,
    steps: list[int] = [48, 64],
    from_pocket: bool = False,
    num_categories: int = 8,
    num_hidden_channels: int = 12,
    channel_dims: list = [42, 42],
    devices: list = [0],
    run_name: str = "atom_channel_reconstruction",
    logging_path: Path | None = None,
    batch_size: int = 4,
    checkpoint: str | None = None,
    **kwargs,
):
    L.seed_everything(SEED, workers=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=logging_path, filename=run_name, save_top_k=5, monitor="loss"
    )
    logger = TensorBoardLogger(logging_path, name=run_name)

    if checkpoint is not None:
        model = LitModel.load_from_checkpoint(checkpoint)
    elif model is None and checkpoint is None:
        model = LitModel(
            normal_std=normal_std,
            alive_threshold=alive_threshold,
            cell_fire_rate=cell_fire_rate,
            num_categories=num_categories,
            num_hidden_channels=num_hidden_channels,
            channel_dims=channel_dims,
            lr=learning_rate,
            steps=steps,
            from_pocket=from_pocket,
        )

    batch_size = min(batch_size, x0.shape[0])
    dataset = torch.utils.data.TensorDataset(x0, target)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )

    trainer = L.Trainer(
        gradient_clip_val=3, 
        gradient_clip_algorithm="value",
        accumulate_grad_batches=1,
        log_every_n_steps=50,
        devices=devices,
        accelerator="gpu",
        max_epochs=num_epochs,
        default_root_dir=os.getcwd(),
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    trainer.fit(model, train_loader)
    return model


def train_loop_ext_force(
    data,
    model: nn.Module | None = None,
    learning_rate: float = 1e-4,
    normal_std: float = 0.01,
    alive_threshold: float = 0.1,
    cell_fire_rate: float = 0.5,
    num_epochs: int = 100,
    steps: list[int] = [48, 49],
    num_categories: int = 8,
    num_hidden_channels: int = 12,
    channel_dims: list = [42, 42],
    logging_path: Path | None = None,
    devices: list = [0],
    run_name: str = "pt_cis_trans_switching",
    batch_size: int = 1,
    checkpoint: str | None = None,
    **kwargs,
    ):
    L.seed_everything(SEED, workers=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=logging_path, filename=run_name, save_top_k=5, monitor="loss"
    )
    logger = TensorBoardLogger(logging_path, name=run_name)

    if checkpoint is not None:
        model = LitModelExtForce.load_from_checkpoint(checkpoint)
    elif model is None and checkpoint is None:
        model = LitModelExtForce(
            normal_std=normal_std,
            alive_threshold=alive_threshold,
            cell_fire_rate=cell_fire_rate,
            num_categories=num_categories,
            num_hidden_channels=num_hidden_channels,
            channel_dims=channel_dims,
            lr=learning_rate,
            steps=steps,
        )

    train_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True, num_workers=1
    )

    trainer = L.Trainer(
        # gradient_clip_val=3, # Does only work if automatic diff is used in lightning
        # gradient_clip_algorithm="value",
        # accumulate_grad_batches=1,
        log_every_n_steps=50,
        devices=devices,
        accelerator="gpu",
        max_epochs=num_epochs,
        default_root_dir=os.getcwd(),
        callbacks=[checkpoint_callback],
        logger=logger,
    )
    trainer.fit(model, train_loader)
    return model
