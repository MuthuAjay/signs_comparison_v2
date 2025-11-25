import os
import sys
import io
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import fitz  # PyMuPDF
import argparse
from tqdm import tqdm


class SignatureDetector:
    """Signature Detection Inference Class"""
    
    def __init__(
        self,
        model_path: str,
        num_classes: int = 4,
        confidence_threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Initialize the signature detector.
        
        Args:
            model_path: Path to the trained model (.pth file)
            num_classes: Number of classes including background
            confidence_threshold: Minimum confidence score for detections
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Class names (customize based on your dataset)
        self.class_names = {
            0: "background",
            1: "signature",
            2: "initials",
            3: "stamp"
        }
        
        # Colors for different classes (BGR format for OpenCV)
        self.colors = {
            1: (0, 255, 0),      # Green for signatures
            2: (255, 0, 0),      # Blue for initials
            3: (0, 0, 255),      # Red for stamps
        }
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the trained model from checkpoint."""
        # Create model architecture
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)

        return model
    
    def preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for inference.
        
        Args:
            image: PIL Image
            
        Returns:
            Transformed image tensor and original size
        """
        original_size = image.size  # (width, height)
        image_tensor = self.transform(image)
        return image_tensor, original_size
    
    @torch.no_grad()
    def detect(self, image: Image.Image) -> List[dict]:
        """
        Perform signature detection on an image.

        Args:
            image: PIL Image

        Returns:
            List of detections with boxes, labels, and scores
        """
        # Preprocess
        image_tensor, original_size = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)

        # Run inference
        predictions = self.model([image_tensor])[0]

        # Filter by confidence threshold
        keep_indices = predictions['scores'] > self.confidence_threshold

        boxes = predictions['boxes'][keep_indices].cpu().numpy()
        labels = predictions['labels'][keep_indices].cpu().numpy()
        scores = predictions['scores'][keep_indices].cpu().numpy()

        # Format detections
        detections = []
        for box, label, score in zip(boxes, labels, scores):
            detections.append({
                'box': box.tolist(),  # [x_min, y_min, x_max, y_max]
                'label': int(label),
                'class_name': self.class_names.get(int(label), 'unknown'),
                'score': float(score)
            })

        return detections

    def filter_signatures_only(self, detections: List[dict]) -> List[dict]:
        """
        Filter detections to keep only signatures (label == 1).

        Args:
            detections: List of all detections

        Returns:
            List of detections containing only signatures
        """
        return [det for det in detections if det['label'] == 1]
    
    def visualize_detections(
        self,
        image: Union[Image.Image, np.ndarray],
        detections: List[dict],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> np.ndarray:
        """
        Visualize detections on the image.
        
        Args:
            image: PIL Image or numpy array
            detections: List of detection dictionaries
            save_path: Optional path to save the visualization
            show: Whether to display the image
            
        Returns:
            Image with drawn bounding boxes as numpy array
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np.copy()
        
        # Draw each detection
        for det in detections:
            box = det['box']
            label = det['label']
            score = det['score']
            class_name = det['class_name']
            
            # Get color for this class
            color = self.colors.get(label, (255, 255, 255))
            
            # Draw bounding box
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw label background
            label_text = f"{class_name}: {score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(
                image_bgr,
                (x_min, y_min - text_height - baseline - 5),
                (x_min + text_width, y_min),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                image_bgr,
                label_text,
                (x_min, y_min - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # Convert back to RGB for display/saving
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, image_bgr)
            print(f"Visualization saved to {save_path}")
        
        # Display if requested
        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(image_rgb)
            plt.axis('off')
            plt.title(f"Detected {len(detections)} signature(s)")
            plt.tight_layout()
            plt.show()
        
        return image_rgb
    
    
    def process_image(
        self,
        image_path: str,
        show: bool = True
    ) -> List[dict]:
        """
        Process a single image file.

        Args:
            image_path: Path to the image file
            show: Whether to display the result

        Returns:
            List of signature detections (filtered to signatures only)
        """
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Detect all objects
        all_detections = self.detect(image)

        # Filter to keep only signatures
        detections = self.filter_signatures_only(all_detections)

        return detections
    
    def process_pdf(
        self,
        pdf_path: str,
        show: bool = True,
        dpi: int = 200
    ) -> List[Tuple[int, List[dict]]]:
        """
        Process a PDF file (all pages) using PyMuPDF.

        Args:
            pdf_path: Path to the PDF file
            show: Whether to display the results
            dpi: DPI for PDF to image conversion (zoom factor)

        Returns:
            List of tuples (page_number, detections)
        """
        # Open PDF with PyMuPDF
        try:
            pdf_document = fitz.open(pdf_path)
        except Exception as e:
            raise

        num_pages = len(pdf_document)

        # Calculate zoom factor from DPI (default PDF is 72 DPI)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        results = []

        for page_num in range(num_pages):
            # Get the page
            page = pdf_document[page_num]

            # Render page to image
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data)).convert('RGB')

            # Detect all objects
            all_detections = self.detect(image)

            # Filter to keep only signatures
            detections = self.filter_signatures_only(all_detections)

            results.append((page_num + 1, detections))

        # Close the PDF
        pdf_document.close()

        return results
    
    def process_file(
        self,
        file_path: str,
        show: bool = True
    ):
        """
        Process either an image or PDF file.

        Args:
            file_path: Path to the file
            show: Whether to display results
        """
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            return self.process_pdf(file_path, show=show)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            return self.process_image(file_path, show=show)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    @staticmethod
    def find_files(folder_path: str, extensions: Optional[List[str]] = None) -> List[str]:
        """
        Recursively find all files with specified extensions in a folder and its subfolders.

        Args:
            folder_path: Root folder path to search
            extensions: List of file extensions to search for (e.g., ['.pdf', '.jpg'])
                       If None, searches for all supported formats

        Returns:
            List of absolute file paths
        """
        if extensions is None:
            # Default: all supported formats
            extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

        # Normalize extensions to lowercase
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                     for ext in extensions]

        found_files = []
        folder_path = Path(folder_path)

        # Walk through directory tree
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_ext = Path(file).suffix.lower()
                if file_ext in extensions:
                    full_path = os.path.join(root, file)
                    found_files.append(full_path)

        return sorted(found_files)
    
    def process_folder(
        self,
        folder_path: str,
        output_dir: Optional[str] = None,
        show: bool = False,
        extensions: Optional[List[str]] = None
    ) -> dict:
        """
        Process all PDFs and images in a folder and its subfolders.

        Args:
            folder_path: Path to the folder containing files
            output_dir: Optional directory to save all results
                       If None, creates 'output' folder in the input folder
            show: Whether to display results (recommended False for batch processing)
            extensions: List of file extensions to process (default: all supported)

        Returns:
            Dictionary containing processing results and statistics
        """
        folder_path = Path(folder_path)
        
        # Setup output directory
        if output_dir is None:
            output_dir = folder_path / "signature_detection_output"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all files
        files = self.find_files(str(folder_path), extensions)

        if not files:
            return {'total_files': 0, 'processed': 0, 'failed': 0, 'results': []}
        
        results = {
            'total_files': len(files),
            'processed': 0,
            'failed': 0,
            'results': []
        }
        
        # Process each file
        for file_path in tqdm(files, desc="Processing files", unit="file"):
            try:
                # Setup paths for this file
                file_ext = Path(file_path).suffix.lower()

                if file_ext == '.pdf':
                    result = self.process_pdf(file_path, show=show)

                    # Get pages with signatures and total count (signatures only)
                    pages_with_signatures = [page_num for page_num, detections in result if len(detections) > 0]
                    total_detections = sum(len(detections) for page_num, detections in result)

                    results['results'].append({
                        'file': str(file_path),
                        'type': 'pdf',
                        'total_pages': len(result),
                        'pages_with_signatures': pages_with_signatures,
                        'total_detections': total_detections,
                        'status': 'success'
                    })

                else:
                    # For images
                    detections = self.process_image(file_path, show=show)

                    results['results'].append({
                        'file': str(file_path),
                        'type': 'image',
                        'detections': len(detections),
                        'status': 'success'
                    })

                results['processed'] += 1

            except Exception as e:
                results['failed'] += 1
                results['results'].append({
                    'file': str(file_path),
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total files: {results['total_files']}")
        print(f"Successfully processed: {results['processed']}")
        print(f"Failed: {results['failed']}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")

        # Save CSV report
        csv_path = output_dir / "detection_results.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['file_path', 'file_type', 'pages', 'has_signatures', 'number_of_signatures'])

            # Write data rows
            for result in results['results']:
                if result['status'] == 'success':
                    file_path = result['file']
                    file_type = result['type']

                    if file_type == 'pdf':
                        # For PDFs, show which pages have signatures as a list
                        pages = str(result['pages_with_signatures']) if result['pages_with_signatures'] else '[]'
                        num_signatures = result['total_detections']
                    else:  # image
                        pages = '[1]' if result['detections'] > 0 else '[]'
                        num_signatures = result['detections']

                    has_signatures = 'Yes' if num_signatures > 0 else 'No'

                    writer.writerow([file_path, file_type, pages, has_signatures, num_signatures])
                else:
                    # For failed files, write with error indication
                    writer.writerow([result['file'], 'error', '[]', 'Error', 0])

        print(f"CSV results saved to: {csv_path}")

        # Save summary report
        summary_path = output_dir / "processing_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("SIGNATURE DETECTION - PROCESSING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Input folder: {folder_path}\n")
            f.write(f"Output folder: {output_dir}\n")
            f.write(f"Total files: {results['total_files']}\n")
            f.write(f"Successfully processed: {results['processed']}\n")
            f.write(f"Failed: {results['failed']}\n\n")

            f.write("DETAILED RESULTS:\n")
            f.write("-" * 60 + "\n")
            for result in results['results']:
                f.write(f"\nFile: {result['file']}\n")
                f.write(f"Status: {result['status']}\n")
                if result['status'] == 'success':
                    if result['type'] == 'pdf':
                        f.write(f"Total pages: {result['total_pages']}\n")
                        f.write(f"Pages with signatures: {result['pages_with_signatures']}\n")
                        f.write(f"Total detections: {result['total_detections']}\n")
                    else:
                        f.write(f"Detections: {result['detections']}\n")
                else:
                    f.write(f"Error: {result.get('error', 'Unknown error')}\n")

        print(f"Summary saved to: {summary_path}")

        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Signature Detection Inference Script - Process folders or individual files"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input folder (will process all PDFs and images recursively) or single file'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.pth file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output directory (for folders) or file (for single images)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the results (recommended for batch processing)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='Device to run inference on'
    )
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=None,
        help='File extensions to process (default: all supported). Example: --extensions .pdf .jpg .png'
    )
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input '{args.input}' not found!")
        sys.exit(1)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        sys.exit(1)
    
    # Initialize detector
    detector = SignatureDetector(
        model_path=args.model,
        confidence_threshold=args.threshold,
        device=args.device
    )
    
    # Process folder or file
    try:
        if os.path.isdir(args.input):
            # Process folder
            detector.process_folder(
                folder_path=args.input,
                output_dir=args.output,
                show=not args.no_show,
                extensions=args.extensions
            )
        else:
            # Process single file
            detector.process_file(
                file_path=args.input,
                show=not args.no_show
            )

    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()