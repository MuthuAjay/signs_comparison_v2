"""
Multi-GPU Signature Detection with DDP (Distributed Data Parallel)
Processes PDFs across multiple GPUs with optimized batching and reporting.
"""

import os
import sys
import csv
import time
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader, DistributedSampler
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import fitz  # PyMuPDF


class PDFPageDataset(IterableDataset):
    """
    Streaming Dataset that reads PDF pages on the fly.
    Optimized for multi-GPU with proper worker splitting.
    """
    def __init__(self, file_paths: List[str], dpi: int = 200, rank: int = 0, world_size: int = 1, files_counter=None):
        self.file_paths = file_paths
        self.dpi = dpi
        self.zoom = dpi / 72
        self.matrix = fitz.Matrix(self.zoom, self.zoom)
        self.rank = rank
        self.world_size = world_size
        self.files_counter = files_counter

    def parse_pdf(self, path: str):
        """Generator that yields pages from a PDF"""
        try:
            doc = fitz.open(path)
            for page_num, page in enumerate(doc):
                # Direct buffer access (no PNG encoding/decoding)
                pix = page.get_pixmap(matrix=self.matrix)

                # Convert raw bytes directly to numpy
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

                # Handle different pixel formats
                if pix.n == 4:  # RGBA
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                elif pix.n == 1:  # Grayscale
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

                yield img_np, path, page_num + 1
            doc.close()

            # Increment completed files counter
            if self.files_counter is not None:
                with self.files_counter.get_lock():
                    self.files_counter.value += 1

        except Exception as e:
            print(f"Error processing {path}: {e}")

    def __iter__(self):
        # Split files across ranks (GPUs)
        per_rank = int(np.ceil(len(self.file_paths) / float(self.world_size)))
        start_idx = self.rank * per_rank
        end_idx = min(start_idx + per_rank, len(self.file_paths))

        rank_files = self.file_paths[start_idx:end_idx]

        for path in rank_files:
            yield from self.parse_pdf(path)


def collate_fn(batch):
    """Custom collator for variable-sized images"""
    images = []
    metadata = []
    transform = transforms.ToTensor()

    for img_np, path, page_num in batch:
        img_tensor = transform(img_np)
        images.append(img_tensor)
        metadata.append({'path': path, 'page': page_num})

    return images, metadata


class SignatureDetectorDDP:
    """Multi-GPU Signature Detection with DDP"""

    def __init__(
        self,
        model_path: str,
        rank: int,
        world_size: int,
        num_classes: int = 4,
        confidence_threshold: float = 0.5,
        batch_size: int = 8,
        use_amp: bool = True
    ):
        self.rank = rank
        self.world_size = world_size
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.device = torch.device(f"cuda:{rank}")

        # Load model
        self.model = self._load_model(model_path)

        # Wrap with DDP
        self.model = DDP(self.model, device_ids=[rank])
        self.model.eval()

        # Class names
        self.class_names = {
            0: "background",
            1: "signature",
            2: "initials",
            3: "stamp"
        }

        if rank == 0:
            print(f"✅ Model loaded with DDP")
            print(f"🔥 World size (GPUs): {world_size}")
            print(f"⚡ Batch size per GPU: {batch_size}")
            print(f"⚡ Mixed Precision (AMP): {use_amp}")

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the trained model from checkpoint."""
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        return model

    def filter_signatures_only(self, detections: List[dict]) -> List[dict]:
        """Filter to keep only signatures (label == 1)"""
        return [det for det in detections if det['label'] == 1]

    @torch.no_grad()
    def process_pdfs_distributed(self, pdf_files: List[str], dpi: int = 200, files_counter=None, total_files: int = 0) -> Dict:
        """
        Process PDFs with DDP across multiple GPUs.
        Each GPU processes its own subset of files.
        """
        local_results = {
            'processed': 0,
            'failed': 0,
            'results': []
        }

        # Create dataset for this rank
        dataset = PDFPageDataset(pdf_files, dpi=dpi, rank=self.rank, world_size=self.world_size, files_counter=files_counter)

        # DataLoader (no DistributedSampler needed for IterableDataset - we handle splitting internally)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2
        )

        # Accumulate results per PDF
        all_pdf_results = {}  # pdf_path -> list of (page_num, detections)

        device_type = 'cuda'

        if self.rank == 0:
            print(f"\n🚀 Starting distributed inference on {self.world_size} GPUs...")
            print(f"📊 Each GPU will show its own progress bar\n")

        # Create progress bar for each rank with unique position
        # Position ensures bars don't overwrite each other
        iterator = tqdm(
            loader,
            desc=f"GPU {self.rank}",
            position=self.rank,
            leave=True,
            unit="batch",
            colour=['green', 'blue', 'yellow', 'red', 'magenta', 'cyan'][self.rank % 6]
        )

        start_time = time.time()
        batch_count = 0
        total_pages = 0

        for images, meta_batch in iterator:
            batch_count += 1
            total_pages += len(images)

            # Move to GPU
            images = [img.to(self.device, non_blocking=True) for img in images]

            # Mixed Precision inference
            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                predictions = self.model(images)

            # Post-process batch
            for pred, meta in zip(predictions, meta_batch):
                pdf_path = meta['path']
                page_num = meta['page']

                if pdf_path not in all_pdf_results:
                    all_pdf_results[pdf_path] = []

                # Filter detections
                scores = pred['scores']
                mask = scores > self.confidence_threshold

                detections = []
                if mask.any():
                    boxes = pred['boxes'][mask]
                    labels = pred['labels'][mask]
                    scores_filtered = scores[mask]

                    for box, label, score in zip(boxes, labels, scores_filtered):
                        detections.append({
                            'box': box.cpu().numpy().tolist(),
                            'label': int(label.cpu().item()),
                            'class_name': self.class_names.get(int(label.cpu().item()), 'unknown'),
                            'score': float(score.cpu().item())
                        })

                # Filter to signatures only
                signature_detections = self.filter_signatures_only(detections)
                all_pdf_results[pdf_path].append((page_num, signature_detections))

            # Update progress bar with speed and remaining files info
            elapsed = time.time() - start_time
            pages_per_sec = total_pages / elapsed if elapsed > 0 else 0

            postfix_dict = {
                'pages': total_pages,
                'pages/s': f'{pages_per_sec:.1f}'
            }

            # Add remaining files count if counter is available
            if files_counter is not None:
                with files_counter.get_lock():
                    completed = files_counter.value
                remaining = total_files - completed
                postfix_dict['remaining'] = remaining

            iterator.set_postfix(postfix_dict)

        # Consolidate results for this rank
        for pdf_path, page_results in all_pdf_results.items():
            page_results.sort(key=lambda x: x[0])

            pages_with_signatures = [pg for pg, dets in page_results if len(dets) > 0]
            total_detections = sum(len(dets) for _, dets in page_results)

            local_results['results'].append({
                'file': str(pdf_path),
                'type': 'pdf',
                'total_pages': len(page_results),
                'pages_with_signatures': pages_with_signatures,
                'total_detections': total_detections,
                'status': 'success'
            })
            local_results['processed'] += 1

        # Print GPU summary
        elapsed_total = time.time() - start_time
        print(f"\n✅ GPU {self.rank} completed:")
        print(f"   Files: {local_results['processed']}")
        print(f"   Total pages: {total_pages}")
        print(f"   Time: {elapsed_total:.1f}s")
        print(f"   Speed: {total_pages/elapsed_total:.1f} pages/s")

        return local_results


def gather_results(local_results: Dict, rank: int, world_size: int) -> Dict:
    """Gather results from all GPUs to rank 0"""
    # Convert to tensors for gathering
    processed = torch.tensor([local_results['processed']], device=f'cuda:{rank}')
    failed = torch.tensor([local_results['failed']], device=f'cuda:{rank}')

    if rank == 0:
        # Gather tensors
        processed_list = [torch.zeros(1, device=f'cuda:{rank}') for _ in range(world_size)]
        failed_list = [torch.zeros(1, device=f'cuda:{rank}') for _ in range(world_size)]

        dist.gather(processed, processed_list, dst=0)
        dist.gather(failed, failed_list, dst=0)

        # Gather results (using object list)
        results_list = [None for _ in range(world_size)]
        dist.gather_object(local_results['results'], results_list, dst=0)

        # Consolidate
        all_results = {
            'total_files': sum(p.item() for p in processed_list) + sum(f.item() for f in failed_list),
            'processed': sum(p.item() for p in processed_list),
            'failed': sum(f.item() for f in failed_list),
            'results': []
        }

        for results in results_list:
            if results:
                all_results['results'].extend(results)

        return all_results
    else:
        # Non-root ranks just send
        dist.gather(processed, dst=0)
        dist.gather(failed, dst=0)
        dist.gather_object(local_results['results'], dst=0)
        return None


def save_results(results: Dict, output_dir: Path, input_folder: Path, batch_size: int, world_size: int, use_amp: bool):
    """Save CSV and summary reports (same format as optimized script)"""
    # Print summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total files: {results['total_files']}")
    print(f"Successfully processed: {results['processed']}")
    print(f"Failed: {results['failed']}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    # Save CSV
    csv_path = output_dir / "detection_results.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_path', 'file_type', 'pages', 'has_signatures', 'number_of_signatures'])

        for result in results['results']:
            if result['status'] == 'success':
                file_path = result['file']
                file_type = result['type']

                if file_type == 'pdf':
                    pages = str(result['pages_with_signatures']) if result['pages_with_signatures'] else '[]'
                    num_signatures = result['total_detections']
                else:
                    pages = '[1]' if result.get('detections', 0) > 0 else '[]'
                    num_signatures = result.get('detections', 0)

                has_signatures = 'Yes' if num_signatures > 0 else 'No'
                writer.writerow([file_path, file_type, pages, has_signatures, num_signatures])
            else:
                writer.writerow([result['file'], 'error', '[]', 'Error', 0])

    print(f"✅ CSV results saved to: {csv_path}")

    # Save summary
    summary_path = output_dir / "processing_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("SIGNATURE DETECTION - PROCESSING SUMMARY (Multi-GPU DDP)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input folder: {input_folder}\n")
        f.write(f"Output folder: {output_dir}\n")
        f.write(f"Number of GPUs: {world_size}\n")
        f.write(f"Batch size per GPU: {batch_size}\n")
        f.write(f"Total batch size: {batch_size * world_size}\n")
        f.write(f"Mixed Precision: {use_amp}\n\n")
        f.write(f"Total files: {results['total_files']}\n")
        f.write(f"Successfully processed: {results['processed']}\n")
        f.write(f"Failed: {results['failed']}\n\n")

        f.write("DETAILED RESULTS:\n")
        f.write("-" * 60 + "\n")
        for result in results['results']:
            f.write(f"\nFile: {result['file']}\n")
            f.write(f"Status: {result['status']}\n")
            if result['status'] == 'success' and result['type'] == 'pdf':
                f.write(f"Total pages: {result['total_pages']}\n")
                f.write(f"Pages with signatures: {result['pages_with_signatures']}\n")
                f.write(f"Total detections: {result['total_detections']}\n")

    print(f"✅ Summary saved to: {summary_path}")


def find_files(folder_path: str, extensions: Optional[List[str]] = None) -> List[str]:
    """Recursively find all files with specified extensions."""
    if extensions is None:
        extensions = ['.pdf']

    extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]

    found_files = []
    folder_path = Path(folder_path)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                found_files.append(os.path.join(root, file))

    return sorted(found_files)


def setup_ddp(rank: int, world_size: int):
    """Initialize DDP environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP"""
    dist.destroy_process_group()


def run_worker(rank: int, world_size: int, args, files_counter, total_files):
    """Worker function for each GPU"""
    setup_ddp(rank, world_size)

    try:
        # Initialize detector
        detector = SignatureDetectorDDP(
            model_path=args.model,
            rank=rank,
            world_size=world_size,
            confidence_threshold=args.threshold,
            batch_size=args.batch_size,
            use_amp=not args.no_amp
        )

        # Find PDF files
        pdf_files = find_files(args.input, ['.pdf'])

        # Process PDFs with shared counter
        local_results = detector.process_pdfs_distributed(
            pdf_files,
            dpi=200,
            files_counter=files_counter,
            total_files=total_files
        )

        # Gather results to rank 0
        all_results = gather_results(local_results, rank, world_size)

        # Save results (only rank 0)
        if rank == 0 and all_results:
            output_dir = Path(args.output) if args.output else Path(args.input) / "signature_detection_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            save_results(all_results, output_dir, Path(args.input), args.batch_size, world_size, not args.no_amp)

    finally:
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU Signature Detection with DDP"
    )
    parser.add_argument('--input', type=str, required=True, help='Input folder with PDFs')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--gpus', type=int, default=None, help='Number of GPUs (default: all available)')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision (AMP)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input '{args.input}' not found!")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        sys.exit(1)

    # Determine number of GPUs
    world_size = args.gpus if args.gpus else torch.cuda.device_count()

    if world_size < 1:
        print("Error: No GPUs available!")
        sys.exit(1)

    # Count total files
    pdf_files = find_files(args.input, ['.pdf'])
    total_files = len(pdf_files)

    print(f"\n{'='*60}")
    print(f"🚀 INITIALIZING MULTI-GPU SIGNATURE DETECTOR")
    print(f"{'='*60}")
    print(f"Total PDF files: {total_files}")
    print(f"GPUs: {world_size}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Total effective batch size: {args.batch_size * world_size}")
    print(f"{'='*60}\n")

    # FIX: Use spawn context for the shared counter (matches mp.spawn)
    ctx = mp.get_context('spawn')
    files_counter = ctx.Value('i', 0)

    # Spawn processes for each GPU
    mp.spawn(
        run_worker,
        args=(world_size, args, files_counter, total_files),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
