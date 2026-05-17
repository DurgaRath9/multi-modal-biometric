"""Main training pipeline entry point.

Usage:
    python train.py
    python train.py training.epochs=10 training.batch_size=16
    python train.py data=default model=default
"""

import argparse
import logging
import sys


def _patch_argparse_for_py314() -> None:
    """Work around Python 3.14 argparse strict help handling.

    Hydra passes a LazyCompletionHelp object as `help=`. Python 3.14's
    argparse performs `"%" in help_string` before coercing to `str`, which
    raises `TypeError` for non-iterable objects.
    """
    if sys.version_info < (3, 14):
        return

    original_expand_help = argparse.HelpFormatter._expand_help

    def _expand_help_compat(self, action):
        help_obj = getattr(action, "help", None)
        if help_obj is not None and not isinstance(help_obj, str):
            action.help = str(help_obj)
        return original_expand_help(self, action)

    argparse.HelpFormatter._expand_help = _expand_help_compat


_patch_argparse_for_py314()

import hydra
import torch
from omegaconf import DictConfig

from src.data.cache import build_metadata_table, get_dataset_summary, save_metadata_cache
from src.data.dataset import MultiModalBiometricDataset
from src.data.kaggle_loader import ensure_dataset
from src.data.loaders import create_dataloaders
from src.data.preprocessing import preprocess_dataset_parallel
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.models.multimodal_model import MultiModalBiometricModel
from src.training.reproducibility import seed_everything
from src.training.trainer import Trainer
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def get_device(device_cfg: str) -> torch.device:
    """Resolve device from config string."""
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()

    logger.info("Starting training pipeline")
    logger.info("Config:\n%s", cfg)

    # Reproducibility
    seed_everything(cfg.seed)

    # Device
    device = get_device(cfg.training.device)
    logger.info("Using device: %s", device)

    # Ensure dataset is available (downloads from Kaggle if enabled)
    data_root = ensure_dataset(
        root_dir=cfg.data.root_dir,
        kaggle_enabled=cfg.data.kaggle.enabled,
        dataset_id=cfg.data.kaggle.dataset_id,
        download_dir=cfg.data.kaggle.get("download_dir"),
    )

    # Build Arrow metadata cache for fast dataset introspection
    metadata_table = build_metadata_table(data_root)
    summary = get_dataset_summary(metadata_table)
    logger.info("Dataset summary: %s", summary)
    save_metadata_cache(metadata_table, "metadata_cache/dataset.parquet")

    # Transforms
    train_tf = get_train_transforms(cfg.data.image_size)
    eval_tf = get_eval_transforms(cfg.data.image_size)

    # Dataset
    dataset = MultiModalBiometricDataset(
        root_dir=data_root,
        transform_iris=train_tf,
        transform_fingerprint=train_tf,
    )

    # Optional Ray preflight: validates all images are readable before training
    if cfg.training.get("use_ray", False):
        import time

        all_paths = [str(s["iris_path"]) for s in dataset.samples if s["iris_path"] is not None] + [
            str(s["fingerprint_path"]) for s in dataset.samples if s["fingerprint_path"] is not None
        ]
        logger.info("Ray preflight: validating %d images in parallel ...", len(all_paths))
        t0 = time.perf_counter()
        preprocess_dataset_parallel(all_paths, target_size=cfg.data.image_size)
        elapsed = time.perf_counter() - t0
        logger.info("Ray preflight complete: %d images validated in %.2fs", len(all_paths), elapsed)

    # DataLoaders
    train_loader, val_loader = create_dataloaders(
        dataset=dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        persistent_workers=cfg.training.persistent_workers,
        prefetch_factor=cfg.training.prefetch_factor,
        train_split=cfg.data.train_split,
        seed=cfg.seed,
    )

    # Model
    model = MultiModalBiometricModel(
        iris_in_channels=cfg.model.iris_encoder.in_channels,
        iris_embedding_dim=cfg.model.iris_encoder.embedding_dim,
        iris_backbone=cfg.model.iris_encoder.backbone,
        iris_pretrained=cfg.model.iris_encoder.pretrained,
        fp_in_channels=cfg.model.fingerprint_encoder.in_channels,
        fp_embedding_dim=cfg.model.fingerprint_encoder.embedding_dim,
        fp_backbone=cfg.model.fingerprint_encoder.backbone,
        fp_pretrained=cfg.model.fingerprint_encoder.pretrained,
        fusion_strategy=cfg.model.fusion.strategy,
        hidden_dim=cfg.model.fusion.hidden_dim,
        num_classes=cfg.model.fusion.num_classes,
        dropout=cfg.model.fusion.dropout,
    )

    # Optimizer & Loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    # LR Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=1e-6
    )

    # Trainer
    checkpoint_dir = (
        cfg.training.checkpoint_dir
        if hasattr(cfg.training, "checkpoint_dir")
        else cfg.checkpoint_dir
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=cfg.log_dir,
        scheduler=scheduler,
        patience=cfg.training.early_stopping_patience,
        use_amp=cfg.training.use_amp,
    )

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.training.epochs,
    )

    logger.info("Training complete. Final metrics: %s", history[-1] if history else "N/A")


if __name__ == "__main__":
    main()
