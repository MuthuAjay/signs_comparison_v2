import os
import sys
import io
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
        
        print(f"Using device: {self.device}")
        
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
        print(f"Loading model from {model_path}...")
        
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
        print("Model loaded successfully!")
        
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
    
    def crop_and_save_detections(
        self,
        image: Union[Image.Image, np.ndarray],
        detections: List[dict],
        output_dir: str,
        prefix: str = "crop",
        padding: int = 10
    ) -> List[str]:
        """
        Crop detected regions and save them as individual images.
        
        Args:
            image: PIL Image or numpy array
            detections: List of detection dictionaries
            output_dir: Directory to save cropped images
            prefix: Prefix for saved filenames
            padding: Extra padding around the bounding box (in pixels)
            
        Returns:
            List of paths to saved crop images
        """
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume RGB numpy array
                image_pil = Image.fromarray(image)
            else:
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        img_width, img_height = image_pil.size
        
        for idx, det in enumerate(detections, 1):
            box = det['box']
            label = det['label']
            class_name = det['class_name']
            score = det['score']
            
            # Extract box coordinates with padding
            x_min, y_min, x_max, y_max = map(int, box)
            
            # Add padding
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(img_width, x_max + padding)
            y_max = min(img_height, y_max + padding)
            
            # Crop the region
            cropped = image_pil.crop((x_min, y_min, x_max, y_max))
            
            # Generate filename
            filename = f"{prefix}_{idx}_{class_name}_{score:.2f}.png"
            save_path = os.path.join(output_dir, filename)
            
            # Save cropped image
            cropped.save(save_path)
            saved_paths.append(save_path)
            
            print(f"  Saved crop {idx}: {save_path}")
        
        return saved_paths
    
    def process_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        show: bool = True,
        save_crops: bool = False,
        crops_dir: Optional[str] = None,
        crop_padding: int = 10
    ) -> Tuple[List[dict], np.ndarray, Optional[List[str]]]:
        """
        Process a single image file.
        
        Args:
            image_path: Path to the image file
            output_path: Optional path to save the result
            show: Whether to display the result
            save_crops: Whether to save cropped detections
            crops_dir: Directory to save crops (defaults to '<image_name>_crops')
            crop_padding: Padding around crops in pixels
            
        Returns:
            Tuple of (detections, visualized image, crop paths)
        """
        print(f"\nProcessing image: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Detect signatures
        detections = self.detect(image)
        
        print(f"Found {len(detections)} detection(s)")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. {det['class_name']} (confidence: {det['score']:.3f})")
        
        # Visualize
        result_image = self.visualize_detections(
            image, detections, save_path=output_path, show=show
        )
        
        # Save crops if requested
        crop_paths = None
        if save_crops and len(detections) > 0:
            if crops_dir is None:
                # Default: create crops directory next to image
                image_stem = Path(image_path).stem
                crops_dir = str(Path(image_path).parent / f"{image_stem}_crops")
            
            print(f"\nSaving {len(detections)} cropped detection(s) to {crops_dir}")
            crop_paths = self.crop_and_save_detections(
                image, detections, crops_dir, 
                prefix=Path(image_path).stem,
                padding=crop_padding
            )
        
        return detections, result_image, crop_paths
    
    def process_pdf(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
        show: bool = True,
        dpi: int = 200,
        save_crops: bool = False,
        crops_dir: Optional[str] = None,
        crop_padding: int = 10
    ) -> List[Tuple[int, List[dict], np.ndarray, Optional[List[str]]]]:
        """
        Process a PDF file (all pages) using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Optional directory to save the results
            show: Whether to display the results
            dpi: DPI for PDF to image conversion (zoom factor)
            save_crops: Whether to save cropped detections
            crops_dir: Directory to save crops (defaults to '<pdf_name>_crops')
            crop_padding: Padding around crops in pixels
            
        Returns:
            List of tuples (page_number, detections, visualized image, crop paths)
        """
        print(f"\nProcessing PDF: {pdf_path}")
        print("Converting PDF to images using PyMuPDF...")
        
        # Open PDF with PyMuPDF
        try:
            pdf_document = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening PDF: {e}")
            print("Make sure PyMuPDF is installed: pip install pymupdf")
            raise
        
        num_pages = len(pdf_document)
        print(f"Processing {num_pages} page(s)...")
        
        # Calculate zoom factor from DPI (default PDF is 72 DPI)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        # Setup crops directory if needed
        if save_crops:
            if crops_dir is None:
                crops_dir = str(Path(pdf_path).parent / f"{Path(pdf_path).stem}_crops")
            os.makedirs(crops_dir, exist_ok=True)
        
        results = []
        
        for page_num in range(num_pages):
            print(f"\n--- Page {page_num + 1}/{num_pages} ---")
            
            # Get the page
            page = pdf_document[page_num]
            
            # Render page to image
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data)).convert('RGB')
            
            # Detect signatures
            detections = self.detect(image)
            
            print(f"Found {len(detections)} detection(s)")
            for i, det in enumerate(detections, 1):
                print(f"  {i}. {det['class_name']} (confidence: {det['score']:.3f})")
            
            # Prepare output path if needed
            save_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(
                    output_dir,
                    f"{Path(pdf_path).stem}_page_{page_num + 1}.jpg"
                )
            
            # Visualize
            result_image = self.visualize_detections(
                image, detections, save_path=save_path, show=show
            )
            
            # Save crops if requested
            crop_paths = None
            if save_crops and len(detections) > 0:
                page_crops_dir = os.path.join(crops_dir, f"page_{page_num + 1}")
                print(f"Saving {len(detections)} cropped detection(s) to {page_crops_dir}")
                crop_paths = self.crop_and_save_detections(
                    image, detections, page_crops_dir,
                    prefix=f"{Path(pdf_path).stem}_p{page_num + 1}",
                    padding=crop_padding
                )
            
            results.append((page_num + 1, detections, result_image, crop_paths))
        
        # Close the PDF
        pdf_document.close()
        
        return results
    
    def process_file(
        self,
        file_path: str,
        output_path: Optional[str] = None,
        show: bool = True,
        save_crops: bool = False,
        crops_dir: Optional[str] = None,
        crop_padding: int = 10
    ):
        """
        Process either an image or PDF file.
        
        Args:
            file_path: Path to the file
            output_path: Optional output path/directory
            show: Whether to display results
            save_crops: Whether to save cropped detections
            crops_dir: Directory to save crops
            crop_padding: Padding around crops in pixels
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return self.process_pdf(
                file_path, output_dir=output_path, show=show,
                save_crops=save_crops, crops_dir=crops_dir, crop_padding=crop_padding
            )
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            return self.process_image(
                file_path, output_path=output_path, show=show,
                save_crops=save_crops, crops_dir=crops_dir, crop_padding=crop_padding
            )
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Signature Detection Inference Script"
    )
    parser.add_argument(
        'input',
        type=str,
        help='Path to input image or PDF file'
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
        help='Path to save output (file for images, directory for PDFs)'
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
        help='Do not display the results'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='Device to run inference on'
    )
    parser.add_argument(
        '--save-crops',
        action='store_true',
        help='Save cropped detections as individual images'
    )
    parser.add_argument(
        '--crops-dir',
        type=str,
        default=None,
        help='Directory to save cropped images (default: auto-generated)'
    )
    parser.add_argument(
        '--crop-padding',
        type=int,
        default=10,
        help='Padding around cropped detections in pixels (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
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
    
    # Process file
    try:
        detector.process_file(
            file_path=args.input,
            output_path=args.output,
            show=not args.no_show,
            save_crops=args.save_crops,
            crops_dir=args.crops_dir,
            crop_padding=args.crop_padding
        )
        print("\n✓ Processing complete!")
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()