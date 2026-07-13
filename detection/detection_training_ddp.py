"""
Multi-GPU Faster R-CNN training for signature detection using
DistributedDataParallel (DDP).

Reuses the dataset, model and per-batch train/eval logic from
detection_training.py so the single-GPU and multi-GPU scripts can't drift
apart; this file only adds the DDP plumbing (process group setup, sampler,
gradient sync, rank-0-only logging/checkpointing).

Usage:
    python detection_training_ddp.py --gpus 4 --batch-size 8 --epochs 20
    # effective batch size = batch-size * gpus
"""

import argparse
import os
from typing import Dict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from detection_training import (
    DEFAULT_CONFIG,
    RANDOM_SEED,
    build_datasets,
    collate_fn,
    evaluate,
    get_model,
    train_one_epoch,
)


def setup_ddp(rank: int, world_size: int, master_port: str):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def all_reduce_mean(value: float, device: torch.device) -> float:
    """Average a Python float across all ranks."""
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor.item()


def build_config(args: argparse.Namespace) -> Dict:
    config = dict(DEFAULT_CONFIG)
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


def run_worker(rank: int, world_size: int, args: argparse.Namespace):
    setup_ddp(rank, world_size, args.master_port)
    is_main = rank == 0
    config = build_config(args)
    device = torch.device(f"cuda:{rank}")
    torch.backends.cudnn.benchmark = True

    try:
        if is_main:
            os.makedirs(config["save_dir"], exist_ok=True)
            print("=" * 60)
            print("Starting Distributed Signature Detection Training")
            print(f"World size (GPUs): {world_size}")
            print(f"Per-GPU batch size: {config['batch_size']}")
            print(f"Effective batch size: {config['batch_size'] * world_size}")
            print("=" * 60)

        # load_annotations_and_split uses a fixed-seed RNG, so every rank
        # independently derives the identical train/test split without
        # needing to broadcast it from rank 0.
        train_dataset, test_dataset, _ = build_datasets(config)

        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=RANDOM_SEED
        )
        test_sampler = DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            sampler=train_sampler,
            num_workers=config["num_workers"],
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=config["num_workers"] > 0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            sampler=test_sampler,
            num_workers=config["num_workers"],
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=config["num_workers"] > 0,
        )

        model = get_model(config["num_classes"]).to(device)
        model = DDP(model, device_ids=[rank])

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(
            params, lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        use_amp = config["use_amp"]
        scaler = torch.amp.GradScaler(enabled=use_amp)

        start_epoch = 0
        best_val_loss = float("inf")
        if config["resume"]:
            if is_main:
                print(f"Resuming from checkpoint: {config['resume']}")
            checkpoint = torch.load(config["resume"], map_location=device)
            model.module.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            best_val_loss = checkpoint.get("val_loss", best_val_loss)
        # Every rank must agree on the starting point before training.
        dist.barrier()

        for epoch in range(start_epoch, config["num_epochs"]):
            train_sampler.set_epoch(epoch)
            if is_main:
                print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
                print("-" * 40)

            local_train_loss = train_one_epoch(
                model,
                optimizer,
                train_loader,
                device,
                epoch + 1,
                num_classes=config["num_classes"],
                grad_clip_norm=config["grad_clip_norm"],
                scaler=scaler if use_amp else None,
                progress=is_main,
            )
            train_loss = all_reduce_mean(local_train_loss, device)

            local_val_loss = evaluate(model, test_loader, device, progress=is_main)
            val_loss = all_reduce_mean(local_val_loss, device)

            # All ranks compute the same averaged loss, so scheduler/optimizer
            # state stays in lockstep without needing to be broadcast.
            scheduler.step(val_loss)

            if is_main:
                print(f"Training loss: {train_loss:.4f}")
                print(f"Validation loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(config["save_dir"], "model_best.pth")
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": model.module.state_dict(),
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
                            "model_state_dict": model.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "config": config,
                        },
                        checkpoint_path,
                    )
                    print(f"Checkpoint saved to {checkpoint_path}")

            # Keep ranks aligned epoch-to-epoch (checkpoint I/O only happens
            # on rank 0 and can otherwise let other ranks race ahead).
            dist.barrier()

        if is_main:
            final_path = os.path.join(config["save_dir"], "model_final.pth")
            torch.save(
                {
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                final_path,
            )
            print(f"\nFinal model saved to {final_path}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print("Training complete!")

    finally:
        cleanup_ddp()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-GPU Faster R-CNN training with DDP")
    parser.add_argument("--data-csv", type=str, default=None, help="Path to annotations CSV")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to image directory")
    parser.add_argument("--image-ids", type=str, default=None, help="Path to image_ids CSV")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Per-GPU batch size (effective = this * gpus)"
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers per GPU")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs (default: all available)")
    parser.add_argument("--master-port", type=str, default="12356", help="DDP master port")
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires at least one CUDA GPU.")

    world_size = args.gpus if args.gpus else torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs available!")

    if world_size == 1:
        print("Only 1 GPU requested/available - running without process spawning overhead "
              "is possible via detection_training.py; continuing with DDP for consistency.")

    mp.spawn(run_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
