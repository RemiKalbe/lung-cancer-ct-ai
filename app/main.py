import app.utils.monkey_patches  # type: ignore # noqa: E401

#

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from typing import Dict, Any

from app.models.unetr import UNETR
from app.data.dataset import LIDCDataset
from app.utils.loss import UNETRLoss
from app.utils.patients import get_patient_ids
from app.data.augmentation import (
    Compose,
    RandomRotation3D,
    RandomFlip3D,
    RandomZoom3D,
    RandomNoise,
    RandomIntensityShift,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_transform(is_train: bool = True):
    """
    Get the transformation pipeline.

    Args:
        is_train (bool): Whether to return transformations for training or validation.

    Returns:
        Compose: A composition of transforms.
    """
    if is_train:
        return Compose(
            [
                RandomRotation3D(max_angle=15),
                RandomFlip3D(),
                RandomZoom3D(min_factor=0.9, max_factor=1.1),
                RandomNoise(noise_variance=0.01),
                RandomIntensityShift(max_shift=0.1),
            ]
        )
    else:
        # For now, no augmentations during validation
        return None


def train(config: Dict[str, Any]):
    """
    Main training function.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing hyperparameters and settings.
    """
    # Set device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Initialize model
    model = UNETR(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
    ).to(device)

    # Initialize loss function and optimizer
    criterion = UNETRLoss(
        segmentation_weight=config["seg_weight"], classification_weight=config["class_weight"]
    )
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # Get patient IDs
    train_ids, val_ids = get_patient_ids(
        max_patients=config["max_patients"], train_ratio=config["train_ratio"], random_selection=False
    )

    # Load datasets
    train_dataset = LIDCDataset(train_ids, transform=config["train_transform"])
    val_dataset = LIDCDataset(val_ids, transform=config["val_transform"])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0
        train_seg_loss = 0.0
        train_class_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            inputs, true_seg, true_class = batch
            inputs, true_seg, true_class = (
                inputs.to(device),
                true_seg.to(device),
                true_class.to(device),
            )

            optimizer.zero_grad()

            pred_seg, pred_class = model(inputs)
            loss, seg_loss, class_loss = criterion(pred_seg, true_seg, pred_class, true_class)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_seg_loss += seg_loss.item()
            train_class_loss += class_loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_seg_loss = 0.0
        val_class_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs, true_seg, true_class = batch
                inputs, true_seg, true_class = (
                    inputs.to(device),
                    true_seg.to(device),
                    true_class.to(device),
                )

                pred_seg, pred_class = model(inputs)
                loss, seg_loss, class_loss = criterion(pred_seg, true_seg, pred_class, true_class)

                val_loss += loss.item()
                val_seg_loss += seg_loss.item()
                val_class_loss += class_loss.item()

        # Log results
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']}:")
        logging.info(f"Train Loss: {train_loss/len(train_loader):.4f}")
        logging.info(f"Train Seg Loss: {train_seg_loss/len(train_loader):.4f}")
        logging.info(f"Train Class Loss: {train_class_loss/len(train_loader):.4f}")
        logging.info(f"Val Loss: {val_loss/len(val_loader):.4f}")
        logging.info(f"Val Seg Loss: {val_seg_loss/len(val_loader):.4f}")
        logging.info(f"Val Class Loss: {val_class_loss/len(val_loader):.4f}")

        # Save checkpoint
        if (epoch + 1) % config["save_frequency"] == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                },
                f"checkpoint_epoch_{epoch+1}.pth",
            )


if __name__ == "__main__":
    # Configuration
    config = {
        "in_channels": 1,
        "out_channels": 2,  # Binary segmentation
        "img_size": (512, 512, 512),
        "patch_size": 16,
        "embed_dim": 768,
        "seg_weight": 1.0,
        "class_weight": 0.5,
        "learning_rate": 1e-4,
        "batch_size": 2,
        "num_epochs": 100,
        "save_frequency": 10,
        "max_patients": 100,  # Set to None to use all patients
        "train_ratio": 0.8,
        "train_transform": get_transform(is_train=True),
        "val_transform": get_transform(is_train=False),
    }

    train(config)
