"""
Example script showing how to use the SignatureDetector class
"""

from inference import SignatureDetector
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================================
# EXAMPLE 1: Basic Image Inference
# ============================================================================

def example_image_inference():
    """Example: Detect signatures in a single image"""
    
    # Initialize detector
    detector = SignatureDetector(
        model_path="models/model_best.pth",  # Path to your trained model
        confidence_threshold=0.5,
        device='cuda'  # or 'cpu'
    )
    
    # Process an image
    image_path = "path/to/your/document.jpg"
    detections, result_image = detector.process_image(
        image_path=image_path,
        output_path="output/result.jpg",  # Optional: save result
        show=True  # Display the result
    )
    
    # Access detection results
    print(f"\nFound {len(detections)} signatures:")
    for i, det in enumerate(detections, 1):
        print(f"{i}. {det['class_name']}: {det['score']:.2%} confidence")
        print(f"   Box: {det['box']}")


# ============================================================================
# EXAMPLE 2: PDF Processing
# ============================================================================

def example_pdf_inference():
    """Example: Detect signatures in all pages of a PDF"""
    
    # Initialize detector
    detector = SignatureDetector(
        model_path="models/model_best.pth",
        confidence_threshold=0.6,
    )
    
    # Process PDF
    pdf_path = "C:\\Users\\EYLAB\\Downloads\\document_55.pdf"
    results = detector.process_pdf(
        pdf_path=pdf_path,
        output_dir="output/pdf_results",  # Save all pages here
        show=True,
        dpi=200  # Quality of PDF to image conversion
    )
    
    # Process results for each page
    for page_num, detections, result_image in results:
        print(f"\nPage {page_num}: {len(detections)} signatures found")


# ============================================================================
# EXAMPLE 3: Batch Processing Multiple Files
# ============================================================================

def example_batch_processing():
    """Example: Process multiple files in a directory"""
    
    import os
    from pathlib import Path
    
    # Initialize detector once
    detector = SignatureDetector(
        model_path="models/model_best.pth",
        confidence_threshold=0.5,
    )
    
    # Directory with images/PDFs
    input_dir = r"C:\Users\EYLAB\Documents\sample_pdfs_\sample_pdfs_"
    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported formats
    image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    pdf_format = '.pdf'
    
    # Process all files
    for file_path in Path(input_dir).iterdir():
        if file_path.suffix.lower() in image_formats:
            output_path = os.path.join(output_dir, f"{file_path.stem}_result.jpg")
            detector.process_image(
                image_path=str(file_path),
                output_path=output_path,
                show=False  # Don't show when batch processing
            )
        elif file_path.suffix.lower() == pdf_format:
            output_subdir = os.path.join(output_dir, file_path.stem)
            detector.process_pdf(
                pdf_path=str(file_path),
                output_dir=output_subdir,
                show=False
            )
    
    print(f"\nBatch processing complete! Results saved to {output_dir}")


# ============================================================================
# EXAMPLE 4: Custom Visualization
# ============================================================================

def example_custom_visualization():
    """Example: Get detections and create custom visualization"""
    
    detector = SignatureDetector(
        model_path="models/model_best.pth",
        confidence_threshold=0.5,
    )
    
    # Load and process image
    image_path = "path/to/your/document.jpg"
    image = Image.open(image_path).convert('RGB')
    
    # Get detections only (no visualization)
    detections = detector.detect(image)
    
    # Create your own custom visualization
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Draw custom bounding boxes
    for det in detections:
        box = det['box']
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        
        # Create rectangle
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=3,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add text
        ax.text(
            x_min, y_min - 10,
            f"{det['class_name']}\n{det['score']:.2%}",
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
            fontsize=10,
            color='black'
        )
    
    ax.axis('off')
    plt.title(f"Custom Visualization - {len(detections)} Signatures Found")
    plt.tight_layout()
    plt.savefig("output/custom_result.jpg", dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# EXAMPLE 5: Filter Detections by Class
# ============================================================================

def example_filter_by_class():
    """Example: Filter detections by specific class"""
    
    detector = SignatureDetector(
        model_path="models/model_best.pth",
        confidence_threshold=0.5,
    )
    
    image_path = "path/to/your/document.jpg"
    image = Image.open(image_path).convert('RGB')
    
    # Get all detections
    detections = detector.detect(image)
    
    # Filter only signatures (class 1)
    signatures_only = [det for det in detections if det['label'] == 1]
    print(f"Total detections: {len(detections)}")
    print(f"Signatures only: {len(signatures_only)}")
    
    # Filter by confidence
    high_confidence = [det for det in detections if det['score'] > 0.8]
    print(f"High confidence (>80%): {len(high_confidence)}")
    
    # Visualize filtered detections
    result_image = detector.visualize_detections(
        image, signatures_only, save_path="output/signatures_only.jpg"
    )


# ============================================================================
# EXAMPLE 6: Extract Signature Regions
# ============================================================================

def example_extract_signatures():
    """Example: Extract and save individual signature regions"""
    
    import os
    import numpy as np
    from PIL import Image
    
    detector = SignatureDetector(
        model_path="models/model_best.pth",
        confidence_threshold=0.7,
    )
    
    image_path = "path/to/your/document.jpg"
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Get detections
    detections = detector.detect(image)
    
    # Create output directory
    output_dir = "output/extracted_signatures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract each signature region
    for i, det in enumerate(detections, 1):
        box = det['box']
        x_min, y_min, x_max, y_max = map(int, box)
        
        # Add padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image_np.shape[1], x_max + padding)
        y_max = min(image_np.shape[0], y_max + padding)
        
        # Crop signature
        signature_crop = image_np[y_min:y_max, x_min:x_max]
        
        # Save
        output_path = os.path.join(
            output_dir,
            f"signature_{i}_{det['class_name']}_{det['score']:.2f}.jpg"
        )
        Image.fromarray(signature_crop).save(output_path)
        print(f"Saved: {output_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Signature Detection - Example Usage")
    print("=" * 60)
    
    # Run examples (uncomment the ones you want to try)
    
    # example_image_inference()
    # example_pdf_inference()
    example_batch_processing()
    # example_custom_visualization()
    # example_filter_by_class()
    # example_extract_signatures()
    
    print("\nTo run examples, uncomment the function calls in the main block")