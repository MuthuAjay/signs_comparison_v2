import argparse
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from tqdm import tqdm

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Default configuration - overridable via CLI (see parse_args / build_config)
DEFAULT_CONFIG = {
    "data_csv_path": r"/home/eyadmin/Documents/Datasets/signs_dataset/full_data.csv",
    "image_dir": r"/home/eyadmin/Documents/Datasets/signs_dataset/images",
    "image_ids": r"/home/eyadmin/Documents/Datasets/signs_dataset/updated_image_ids.csv",
    "train_split": 0.8,
    "batch_size": 8,
    "num_epochs": 2,
    "learning_rate": 0.0001,
    "weight_decay": 1e-5,
    "save_dir": "models",
    "input_size": (512, 512),
    "num_classes": 4,  # Background is class 0, actual classes are 1-3
    "confidence_threshold": 0.5,
    "num_workers": 4,
    "grad_clip_norm": 10.0,
    "use_amp": True,
    "resume": None,  # path to a checkpoint to resume from
}


class FixedSignatureDataset(Dataset):
    """Dataset class for signature detection with proper validation and error handling."""

    def __init__(
        self,
        full_data_df: pd.DataFrame,
        img_dir: str,
        id_to_file: Dict[str, str],
        image_ids: Optional[List] = None,
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (512, 512),
        num_classes: int = 4,
    ):
        """
        Initialize SignatureDataset with validation.

        Args:
            full_data_df: DataFrame with all annotations
            img_dir: Directory with images
            id_to_file: Dictionary mapping image IDs to filenames
            image_ids: List of image IDs to use (for train/test split)
            transform: Optional transform to be applied on a sample
            target_size: Size to resize images to (height, width)
            num_classes: Number of classes including background (class 0)
        """
        self.full_data_df = full_data_df
        self.img_dir = img_dir
        self.transform = transform
        self.target_size = target_size
        self.id_to_file = id_to_file
        self.num_classes = num_classes

        # If image_ids are provided, use only those
        if image_ids is not None:
            self.image_ids = list(image_ids)
        else:
            self.image_ids = list(full_data_df["image_id"].unique())

        # Group by image_id once (O(n)) instead of filtering the full
        # dataframe per image (O(n_images * n_rows)).
        groups = dict(tuple(full_data_df.groupby("image_id")))
        empty = full_data_df.iloc[0:0]
        self.img_to_annotations = {
            img_id: groups.get(img_id, empty) for img_id in self.image_ids
        }

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, self.id_to_file[str(img_id)])

        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size

        # Resize
        image = image.resize(self.target_size[::-1])  # PIL expects (width, height)

        # Get all bounding boxes for this image
        img_annotations = self.img_to_annotations[img_id]

        boxes = []
        labels = []

        for row in img_annotations.itertuples(index=False):
            bbox = row.bbox

            # Convert normalized coordinates to absolute coordinates
            x_min = bbox[0] * orig_width
            y_min = bbox[1] * orig_height
            width = bbox[2] * orig_width
            height = bbox[3] * orig_height

            # Convert to [x_min, y_min, x_max, y_max] format
            x_max = x_min + width
            y_max = y_min + height

            # Scale coordinates to target size
            x_min = x_min * self.target_size[1] / orig_width
            y_min = y_min * self.target_size[0] / orig_height
            x_max = x_max * self.target_size[1] / orig_width
            y_max = y_max * self.target_size[0] / orig_height

            # Skip invalid boxes
            if x_max <= x_min or y_max <= y_min or x_min < 0 or y_min < 0:
                continue

            # Fix category_id to ensure it's valid (0 < label < num_classes)
            category_id = row.category_id

            # If category_id is invalid, adjust it or skip
            if category_id < 0 or category_id >= self.num_classes:
                print(
                    f"Warning: Invalid category_id {category_id} found, adjusting to valid range"
                )
                # Clamp to valid range
                category_id = max(1, min(category_id, self.num_classes - 1))

            # Add to lists
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(category_id)

        # Handle case with no valid boxes
        if len(boxes) == 0:
            # Skip images with no valid annotations by using minimal valid box
            # Using class 1 instead of 0 (background shouldn't be in training)
            boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
            labels = torch.tensor([1], dtype=torch.long)
            print(f"Warning: Image {img_id} has no valid boxes, using placeholder")
        else:
            # Convert to tensors
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        return image, target


def collate_fn(batch):
    """Custom collate function for variable-sized batches."""
    return tuple(zip(*batch))


def load_annotations_and_split(
    config: Dict,
) -> Tuple[pd.DataFrame, Dict[str, str], np.ndarray, np.ndarray]:
    """
    Load annotation/image-id CSVs and produce a reproducible train/test split
    of image ids. Shared by both the single-process and DDP training scripts
    so the split is identical regardless of how many processes/GPUs load it.

    Returns:
        full_data_df, id_to_file, train_image_ids, test_image_ids
    """
    print("Loading data...")
    full_data_df = pd.read_csv(config["data_csv_path"])
    # bbox is stored as a stringified list in the CSV; parse it once here
    # instead of re-parsing on every __getitem__ call.
    import ast

    full_data_df = full_data_df.copy()
    full_data_df["bbox"] = full_data_df["bbox"].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else v
    )
    print(f"Loaded {len(full_data_df)} annotations")

    # Load image ID to file name mapping
    img_id_meta = pd.read_csv(config["image_ids"])
    id_to_file = {
        str(id_val): file_name
        for id_val, file_name in zip(img_id_meta["id"], img_id_meta["file_name"])
    }

    # Get unique image IDs, dropping any without a known file mapping so we
    # fail fast here instead of a KeyError deep inside a DataLoader worker.
    unique_image_ids = full_data_df["image_id"].unique()
    known_mask = np.array([str(i) in id_to_file for i in unique_image_ids])
    if not known_mask.all():
        missing = (~known_mask).sum()
        print(f"Warning: dropping {missing} image_ids with no entry in image_ids CSV")
        unique_image_ids = unique_image_ids[known_mask]
    print(f"Total usable images: {len(unique_image_ids)}")

    # Shuffle and split image IDs with a dedicated RNG (independent of any
    # other code that touches the global numpy random state).
    rng = np.random.default_rng(RANDOM_SEED)
    unique_image_ids = unique_image_ids.copy()
    rng.shuffle(unique_image_ids)

    train_size = int(len(unique_image_ids) * config["train_split"])
    train_image_ids = unique_image_ids[:train_size]
    test_image_ids = unique_image_ids[train_size:]

    return full_data_df, id_to_file, train_image_ids, test_image_ids


def build_datasets(
    config: Dict,
) -> Tuple[FixedSignatureDataset, FixedSignatureDataset, Dict[str, str]]:
    """Build train/test datasets (no DataLoaders — callers may want a
    DistributedSampler instead of shuffle=True)."""
    full_data_df, id_to_file, train_image_ids, test_image_ids = load_annotations_and_split(
        config
    )

    data_transforms = transforms.Compose([transforms.ToTensor()])

    train_dataset = FixedSignatureDataset(
        full_data_df,
        config["image_dir"],
        id_to_file,
        image_ids=train_image_ids,
        transform=data_transforms,
        target_size=config["input_size"],
        num_classes=config["num_classes"],
    )

    test_dataset = FixedSignatureDataset(
        full_data_df,
        config["image_dir"],
        id_to_file,
        image_ids=test_image_ids,
        transform=data_transforms,
        target_size=config["input_size"],
        num_classes=config["num_classes"],
    )

    print(f"Training set: {len(train_dataset)} images")
    print(f"Test set: {len(test_dataset)} images")

    return train_dataset, test_dataset, id_to_file


def load_data_and_prepare_datasets(config: Dict) -> Tuple[DataLoader, DataLoader, Dict[str, str]]:
    """Load and prepare train/test DataLoaders with proper splitting."""
    train_dataset, test_dataset, id_to_file = build_datasets(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config["num_workers"] > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config["num_workers"] > 0,
    )

    return train_loader, test_loader, id_to_file


def get_model(num_classes: int) -> torch.nn.Module:
    """
    Create Faster R-CNN model with custom number of classes.

    Args:
        num_classes: Number of classes including background

    Returns:
        Faster R-CNN model
    """
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the classifier head for our custom number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def _validate_targets(targets, num_classes: int) -> bool:
    """Return True if all targets have labels in the valid [1, num_classes) range."""
    for t in targets:
        if (t["labels"] >= num_classes).any() or (t["labels"] < 1).any():
            print(
                f"Warning: Invalid label values in batch: {t['labels'].tolist()} "
                f"(valid range: 1 to {num_classes - 1})"
            )
            return False
    return True


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    num_classes: int,
    grad_clip_norm: float = 10.0,
    scaler: Optional[torch.amp.GradScaler] = None,
    progress: bool = True,
) -> float:
    """
    Train model for one epoch.

    Args:
        model: The detection model
        optimizer: Optimizer for training
        data_loader: Training data loader
        device: Device to train on
        epoch: Current epoch number
        num_classes: Number of classes (including background) for label validation
        grad_clip_norm: Max norm for gradient clipping
        scaler: Optional torch.amp.GradScaler for mixed-precision training
        progress: Whether to render a tqdm progress bar (disable on non-zero DDP ranks)

    Returns:
        Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None
    device_type = "cuda" if device.type == "cuda" else "cpu"

    iterator = tqdm(data_loader, desc=f"Epoch {epoch}") if progress else data_loader
    for images, targets in iterator:
        try:
            if not _validate_targets(targets, num_classes):
                continue

            # Move data to device
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets
            ]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                print(f"Warning: Non-finite loss detected: {losses.item()}, skipping batch")
                continue

            if use_amp:
                scaler.scale(losses).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            running_loss += losses.item()
            num_batches += 1
            if progress:
                iterator.set_postfix({"loss": f"{losses.item():.4f}"})

        except RuntimeError as e:
            print(f"Runtime error in batch: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error in batch: {type(e).__name__}: {e}")
            continue

    avg_loss = running_loss / num_batches if num_batches > 0 else float("inf")
    return avg_loss


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    progress: bool = True,
) -> float:
    """
    Evaluate model on validation set.

    Note: torchvision detection models only return a loss dict in train()
    mode, so we keep the model in train mode here (under no_grad, so no
    gradients are computed/no weights updated). The ResNet-FPN backbone uses
    FrozenBatchNorm2d, so BatchNorm running stats are not affected.

    Args:
        model: The detection model
        data_loader: Validation data loader
        device: Device to evaluate on
        progress: Whether to render a tqdm progress bar (disable on non-zero DDP ranks)

    Returns:
        Average loss on validation set
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    iterator = tqdm(data_loader, desc="Validation") if progress else data_loader
    for images, targets in iterator:
        try:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            running_loss += losses.item()
            num_batches += 1
            if progress:
                iterator.set_postfix({"loss": f"{losses.item():.4f}"})

        except Exception as e:
            print(f"Error in validation batch: {e}")
            continue

    avg_loss = running_loss / num_batches if num_batches > 0 else float("inf")
    return avg_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Faster R-CNN signature detector")
    parser.add_argument("--data-csv", type=str, default=None, help="Path to annotations CSV")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to image directory")
    parser.add_argument("--image-ids", type=str, default=None, help="Path to image_ids CSV")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    return parser.parse_args()


def build_config(args: Optional[argparse.Namespace] = None) -> Dict:
    config = dict(DEFAULT_CONFIG)
    if args is None:
        return config

    overrides = {
        "data_csv_path": args.data_csv,
        "image_dir": args.image_dir,
        "image_ids": args.image_ids,
        "save_dir": args.save_dir,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_workers": args.num_workers,
        "resume": args.resume,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.no_amp:
        config["use_amp"] = False
    return config


def main():
    """Main training function."""
    args = parse_args()
    config = build_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        # Input size is fixed (config["input_size"]), so cudnn can safely
        # cache the best convolution algorithms for that shape.
        torch.backends.cudnn.benchmark = True

    os.makedirs(config["save_dir"], exist_ok=True)

    print("=" * 60)
    print("Starting Signature Detection Training")
    print("=" * 60)

    # Load data
    train_loader, test_loader, id_to_file = load_data_and_prepare_datasets(config)

    # Create model
    print("\nInitializing model...")
    model = get_model(config["num_classes"])
    model.to(device)
    print(f"Model loaded on {device}")

    # Optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        params,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    use_amp = config["use_amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    best_val_loss = float("inf")
    if config["resume"]:
        print(f"Resuming from checkpoint: {config['resume']}")
        checkpoint = torch.load(config["resume"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("val_loss", best_val_loss)

    # Training loop
    print("\nStarting training...")
    print(f"Total epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Mixed precision: {use_amp}")
    print("=" * 60)

    for epoch in range(start_epoch, config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 40)

        try:
            train_loss = train_one_epoch(
                model,
                optimizer,
                train_loader,
                device,
                epoch + 1,
                num_classes=config["num_classes"],
                grad_clip_norm=config["grad_clip_norm"],
                scaler=scaler if use_amp else None,
            )
            print(f"Training loss: {train_loss:.4f}")

            val_loss = evaluate(model, test_loader, device)
            print(f"Validation loss: {val_loss:.4f}")

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(config["save_dir"], "model_best.pth")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "config": config,
                    },
                    best_path,
                )
                print(f"Best model saved to {best_path} (val_loss: {val_loss:.4f})")

            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(
                    config["save_dir"], f"checkpoint_epoch_{epoch + 1}.pth"
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "config": config,
                    },
                    checkpoint_path,
                )
                print(f"Checkpoint saved to {checkpoint_path}")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            break
        except Exception as e:
            print(f"Error during epoch {epoch + 1}: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    final_path = os.path.join(config["save_dir"], "model_final.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        final_path,
    )
    print(f"Final model saved to {final_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
