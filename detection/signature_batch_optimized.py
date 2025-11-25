import os
import sys
import csv
from typing import List, Tuple, Union, Optional
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import fitz  # PyMuPDF
import argparse
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader


class PDFPageDataset(IterableDataset):
    """
    Streaming Dataset that reads PDF pages on the fly.
    Prevents OOM and enables multi-process loading.
    """
    def __init__(self, file_paths: List[str], dpi: int = 200):
        self.file_paths = file_paths
        self.dpi = dpi
        self.zoom = dpi / 72
        self.matrix = fitz.Matrix(self.zoom, self.zoom)

    def parse_pdf(self, path: str):
        """Generator that yields pages from a PDF"""
        doc = fitz.open(path)
        for page_num, page in enumerate(doc):
            # OPTIMIZATION 1: Direct buffer access (No PNG encoding/decoding)
            pix = page.get_pixmap(matrix=self.matrix)
            
            # Convert raw bytes directly to numpy (H, W, channels)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            # Handle different pixel formats
            if pix.n == 4:  # RGBA
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            elif pix.n == 1:  # Grayscale
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            # else: already RGB (pix.n == 3)

            yield img_np, path, page_num + 1
        doc.close()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single-process data loading
            for path in self.file_paths:
                yield from self.parse_pdf(path)
        else:
            # Multi-process: Split files across workers
            per_worker = int(np.ceil(len(self.file_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_paths))
            
            for i in range(iter_start, iter_end):
                yield from self.parse_pdf(self.file_paths[i])


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


class SignatureDetectorOptimized:
    """Optimized Signature Detection with Streaming + Mixed Precision + Multi-processing"""
    
    def __init__(
        self,
        model_path: str,
        num_classes: int = 4,
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
        batch_size: int = 16,  # Increased default
        num_workers: int = 4,
        use_amp: bool = True,
        use_compile: bool = True
    ):
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_amp = use_amp

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

        # OPTIMIZATION: Compile model (PyTorch 2.0+)
        if use_compile and hasattr(torch, 'compile') and self.device.type == 'cuda':
            print("⚡ Compiling model with torch.compile...")
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("✅ Model compiled successfully")
            except Exception as e:
                print(f"⚠️ Compilation failed: {e}. Continuing without compilation.")

        # Class names
        self.class_names = {
            0: "background",
            1: "signature",
            2: "initials",
            3: "stamp"
        }

        print(f"✅ Model loaded on {self.device}")
        print(f"⚡ Batch size: {self.batch_size}")
        print(f"⚡ Num workers: {self.num_workers}")
        print(f"⚡ Mixed Precision (AMP): {self.use_amp}")
    
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
    def process_pdfs_optimized(self, pdf_files: List[str], dpi: int = 200) -> dict:
        """
        Optimized PDF processing with streaming + DataLoader + AMP.
        
        Key improvements:
        1. Streaming pages (no memory explosion)
        2. Multi-process data loading (CPU stays busy)
        3. Mixed precision inference (2x faster on Ada cards)
        """
        batch_results = {
            'processed': 0,
            'failed': 0,
            'results': []
        }

        # Create streaming dataset
        dataset = PDFPageDataset(pdf_files, dpi=dpi)
        
        # OPTIMIZATION 2: DataLoader with workers for parallel CPU processing
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            collate_fn=collate_fn, 
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )

        # Accumulate results per PDF
        all_pdf_results = {}  # pdf_path -> list of (page_num, detections)
        
        # Determine device type for AMP
        device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
        
        print(f"🚀 Starting optimized inference...")
        
        for images, meta_batch in tqdm(loader, desc="Processing batches", unit="batch"):
            # Move to GPU
            images = [img.to(self.device, non_blocking=True) for img in images]
            
            # OPTIMIZATION 3: Automatic Mixed Precision
            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                predictions = self.model(images)

            # Post-process batch
            for pred, meta in zip(predictions, meta_batch):
                pdf_path = meta['path']
                page_num = meta['page']
                
                # Initialize PDF entry if needed
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
                    
                    # Format detections
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

        # Consolidate results
        for pdf_path, page_results in all_pdf_results.items():
            page_results.sort(key=lambda x: x[0])
            
            pages_with_signatures = [pg for pg, dets in page_results if len(dets) > 0]
            total_detections = sum(len(dets) for _, dets in page_results)
            
            batch_results['results'].append({
                'file': str(pdf_path),
                'type': 'pdf',
                'total_pages': len(page_results),
                'pages_with_signatures': pages_with_signatures,
                'total_detections': total_detections,
                'status': 'success'
            })
            batch_results['processed'] += 1

        return batch_results

    @staticmethod
    def find_files(folder_path: str, extensions: Optional[List[str]] = None) -> List[str]:
        """Recursively find all files with specified extensions."""
        if extensions is None:
            extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                     for ext in extensions]

        found_files = []
        folder_path = Path(folder_path)

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if Path(file).suffix.lower() in extensions:
                    found_files.append(os.path.join(root, file))

        return sorted(found_files)
    
    def process_folder(
        self,
        folder_path: str,
        output_dir: Optional[str] = None,
        extensions: Optional[List[str]] = None
    ) -> dict:
        """Process all PDFs and images in a folder with optimizations."""
        folder_path = Path(folder_path)

        if output_dir is None:
            output_dir = folder_path / "signature_detection_output"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Find files
        files = self.find_files(str(folder_path), extensions)
        
        if not files:
            return {'total_files': 0, 'processed': 0, 'failed': 0, 'results': []}

        results = {
            'total_files': len(files),
            'processed': 0,
            'failed': 0,
            'results': []
        }

        # Separate PDFs and images
        pdf_files = [f for f in files if Path(f).suffix.lower() == '.pdf']
        image_files = [f for f in files if Path(f).suffix.lower() in 
                      ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']]

        # Process PDFs with optimized pipeline
        if pdf_files:
            print(f"\n📄 Processing {len(pdf_files)} PDF files with optimized pipeline...")
            try:
                pdf_results = self.process_pdfs_optimized(pdf_files)
                results['results'].extend(pdf_results['results'])
                results['processed'] += pdf_results['processed']
                results['failed'] += pdf_results['failed']
            except Exception as e:
                print(f"Error processing PDFs: {e}")
                import traceback
                traceback.print_exc()

        # TODO: Add optimized image processing if needed
        
        # Save results
        self._save_results(results, output_dir, folder_path)
        
        return results

    def _save_results(self, results: dict, output_dir: Path, input_folder: Path):
        """Save CSV and summary reports."""
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
            f.write("SIGNATURE DETECTION - PROCESSING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Input folder: {input_folder}\n")
            f.write(f"Output folder: {output_dir}\n")
            f.write(f"Batch size: {self.batch_size}\n")
            f.write(f"Num workers: {self.num_workers}\n")
            f.write(f"Mixed Precision: {self.use_amp}\n\n")
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


def main():
    parser = argparse.ArgumentParser(
        description="Optimized Signature Detection - Streaming + DataLoader + AMP"
    )
    parser.add_argument('--input', type=str, required=True, 
                       help='Input folder with PDFs/images')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model (.pth)')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda:2', 
                       help='Device (cuda, cuda:0, cuda:1, etc.)')
    parser.add_argument('--batch-size', type=int, default=16, 
                       help='Batch size (increase for more GPU utilization)')
    parser.add_argument('--num-workers', type=int, default=4, 
                       help='Number of CPU workers for data loading')
    parser.add_argument('--no-amp', action='store_true', 
                       help='Disable mixed precision (AMP)')
    parser.add_argument('--no-compile', action='store_true', 
                       help='Disable torch.compile')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input '{args.input}' not found!")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"🚀 INITIALIZING OPTIMIZED SIGNATURE DETECTOR")
    print(f"{'='*60}")
    
    detector = SignatureDetectorOptimized(
        model_path=args.model,
        confidence_threshold=args.threshold,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        use_compile=not args.no_compile
    )
    
    print(f"{'='*60}\n")
    
    try:
        detector.process_folder(
            folder_path=args.input,
            output_dir=args.output
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()