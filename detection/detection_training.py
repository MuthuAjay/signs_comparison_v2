import os
import ast
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

# Configuration - Define before use
CONFIG = {
    "data_csv_path": r"C:\Users\EYLAB\Downloads\signver\full_data.csv",
    "image_dir": r"C:\Users\EYLAB\Downloads\signver\images",
    "image_ids": r"C:\Users\EYLAB\Downloads\signver\updated_image_ids.csv",
    "train_split": 0.8,
    "batch_size": 8,
    "num_epochs": 2,
    "learning_rate": 0.0001,
    "weight_decay": 1e-5,
    "save_dir": "models",
    "input_size": (512, 512),
    "num_classes": 4,  # Background is class 0, actual classes are 1-3
    "confidence_threshold": 0.5,
    "num_workers": 0,  # Set to 0 for Windows, increase on Linux
}

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create save directory
os.makedirs(CONFIG["save_dir"], exist_ok=True)


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
            self.image_ids = image_ids
        else:
            self.image_ids = full_data_df["image_id"].unique()

        # Group by image_id to get all annotations for each image
        self.img_to_annotations = {
            img_id: self.full_data_df[self.full_data_df["image_id"] == img_id]
            for img_id in self.image_ids
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

        for _, row in img_annotations.iterrows():
            # Parse bbox from string to list if needed
            if isinstance(row["bbox"], str):
                bbox = ast.literal_eval(row["bbox"])
            else:
                bbox = row["bbox"]

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
            category_id = row["category_id"]

            # If category_id is invalid, adjust it or skip
            if category_id < 0 or category_id >= self.num_classes:
                print(
                    f"Warning: Invalid category_id {category_id} found, adjusting to valid range"
                )
                # Option 1: Skip this annotation
                # continue

                # Option 2: Clamp to valid range
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


def load_data_and_prepare_datasets() -> Tuple[DataLoader, DataLoader, Dict[str, str]]:
    """Load and prepare train/test datasets with proper splitting."""
    # Load the full data
    print("Loading data...")
    full_data_df = pd.read_csv(CONFIG["data_csv_path"])
    print(f"Loaded {len(full_data_df)} annotations")

    # Load image ID to file name mapping
    img_id_meta = pd.read_csv(CONFIG["image_ids"])
    id_to_file = {
        str(id_val): file_name
        for id_val, file_name in zip(img_id_meta["id"], img_id_meta["file_name"])
    }

    # Get unique image IDs
    unique_image_ids = full_data_df["image_id"].unique()
    print(f"Total unique images: {len(unique_image_ids)}")

    # Split into train and test sets
    train_size = int(len(unique_image_ids) * CONFIG["train_split"])

    # Shuffle and split image IDs (using seeded random for reproducibility)
    np.random.shuffle(unique_image_ids)
    train_image_ids = unique_image_ids[:train_size]
    test_image_ids = unique_image_ids[train_size:]

    # Data transformations
    data_transforms = transforms.Compose([transforms.ToTensor()])

    # Create datasets
    train_dataset = FixedSignatureDataset(
        full_data_df,
        CONFIG["image_dir"],
        id_to_file,
        image_ids=train_image_ids,
        transform=data_transforms,
        target_size=CONFIG["input_size"],
        num_classes=CONFIG["num_classes"],
    )

    test_dataset = FixedSignatureDataset(
        full_data_df,
        CONFIG["image_dir"],
        id_to_file,
        image_ids=test_image_ids,
        transform=data_transforms,
        target_size=CONFIG["input_size"],
        num_classes=CONFIG["num_classes"],
    )

    print(f"Training set: {len(train_dataset)} images")
    print(f"Test set: {len(test_dataset)} images")

    # Create data loaders with proper collate function
    def collate_fn(batch):
        """Custom collate function for variable-sized batches."""
        return tuple(zip(*batch))

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
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


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Train model for one epoch.

    Args:
        model: The detection model
        optimizer: Optimizer for training
        data_loader: Training data loader
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}")
    for images, targets in progress_bar:
        try:
            # Validate labels before processing
            invalid_batch = False
            for t in targets:
                if (t["labels"] >= CONFIG["num_classes"]).any() or (t["labels"] < 1).any():
                    print(
                        f"Warning: Invalid label values in batch: {t['labels'].tolist()} "
                        f"(valid range: 1 to {CONFIG['num_classes']-1})"
                    )
                    invalid_batch = True
                    break

            if invalid_batch:
                continue

            # Move data to device
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Check for invalid loss
            if not torch.isfinite(losses):
                print(f"Warning: Non-finite loss detected: {losses.item()}, skipping batch")
                continue

            # Backward pass and optimize
            optimizer.zero_grad()
            losses.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()

            # Update metrics
            running_loss += losses.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{losses.item():.4f}"})

        except RuntimeError as e:
            print(f"Runtime error in batch: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error in batch: {type(e).__name__}: {e}")
            continue

    avg_loss = running_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate model on validation set.

    Args:
        model: The detection model
        data_loader: Validation data loader
        device: Device to evaluate on

    Returns:
        Average loss on validation set
    """
    model.train()  # Keep in train mode to compute loss
    running_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(data_loader, desc="Validation")
    for images, targets in progress_bar:
        try:
            # Move data to device
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass only
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            running_loss += losses.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{losses.item():.4f}"})

        except Exception as e:
            print(f"Error in validation batch: {e}")
            continue

    avg_loss = running_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss


def main():
    """Main training function."""
    # Set CUDA debugging for better error messages
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    print("="*60)
    print("Starting Signature Detection Training")
    print("="*60)

    # Load data
    train_loader, test_loader, id_to_file = load_data_and_prepare_datasets()

    # Create model
    print("\nInitializing model...")
    model = get_model(CONFIG["num_classes"])
    model.to(device)
    print(f"Model loaded on {device}")

    # Optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        params,
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Training loop
    print("\nStarting training...")
    print(f"Total epochs: {CONFIG['num_epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print("="*60)

    best_val_loss = float('inf')

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        print("-" * 40)

        try:
            # Train one epoch
            train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch+1)
            print(f"Training loss: {train_loss:.4f}")

            # Validate
            val_loss = evaluate(model, test_loader, device)
            print(f"Validation loss: {val_loss:.4f}")

            # Update learning rate based on validation loss
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(CONFIG["save_dir"], "model_best.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, best_path)
                print(f"Best model saved to {best_path} (val_loss: {val_loss:.4f})")

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(
                    CONFIG["save_dir"], f"checkpoint_epoch_{epoch+1}.pth"
                )
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            break
        except Exception as e:
            print(f"Error during epoch {epoch+1}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save final model
    print("\n" + "="*60)
    final_path = os.path.join(CONFIG["save_dir"], "model_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': CONFIG,
    }, final_path)
    print(f"Final model saved to {final_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
