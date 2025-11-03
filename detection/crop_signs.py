"""
Example usage of the enhanced SignatureDetector with crop functionality
"""

from inference import SignatureDetector

# Initialize detector
detector = SignatureDetector(
    model_path=r"C:\Users\EYLAB\Documents\GitHub\signs_comparison_v2\models\model_best.pth",
    confidence_threshold=0.5
)

# ============================================
# Example 1: Process image with crops saved
# ============================================
# detections, result_img, crop_paths = detector.process_image(
#     image_path="document.jpg",
#     output_path="result_with_boxes.jpg",  # Save annotated image
#     show=True,                             # Display the result
#     save_crops=True,                       # Enable crop saving
#     crops_dir="extracted_signatures",      # Custom crops directory
#     crop_padding=15                        # Add 15px padding around each crop
# )

# # Print crop paths
# if crop_paths:
#     print(f"\nSaved {len(crop_paths)} cropped signatures:")
#     for path in crop_paths:
#         print(f"  - {path}")


# ============================================
# Example 2: Process PDF with crops saved
# ============================================
results = detector.process_pdf(
    pdf_path=r"C:\Users\EYLAB\Documents\sample_pdfs_\sample_pdfs_\Handwriting samples of Jen Chiu Kao Denis.pdf",
    output_dir="org_file",          # Directory for annotated pages
    show=False,                        # Don't display (useful for batch processing)
    dpi=200,                           # High quality conversion
    save_crops=True,                   # Enable crop saving
    crops_dir="org_file_pdf_signatures",        # All crops go here
    crop_padding=20                    # More padding for PDF crops
)

# Process results
total_signatures = 0
for page_num, detections, result_img, crop_paths in results:
    print(f"\nPage {page_num}: Found {len(detections)} signatures")
    total_signatures += len(detections)
    if crop_paths:
        for path in crop_paths:
            print(f"  - {path}")

print(f"\nTotal signatures extracted: {total_signatures}")

results = detector.process_pdf(
    pdf_path=r"C:\Users\EYLAB\Documents\sample_pdfs_\sample_pdfs_\Disputed documents.pdf",
    output_dir="disp_file",          # Directory for annotated pages
    show=False,                        # Don't display (useful for batch processing)
    dpi=200,                           # High quality conversion
    save_crops=True,                   # Enable crop saving
    crops_dir="disp_file_pdf_signatures",        # All crops go here
    crop_padding=20                    # More padding for PDF crops
)

# Process results
total_signatures = 0
for page_num, detections, result_img, crop_paths in results:
    print(f"\nPage {page_num}: Found {len(detections)} signatures")
    total_signatures += len(detections)
    if crop_paths:
        for path in crop_paths:
            print(f"  - {path}")

print(f"\nTotal signatures extracted: {total_signatures}")



# ============================================
# Example 3: Process from command line
# ============================================
"""
# Save cropped signatures from an image
python signature_detector_enhanced.py input.jpg \
    --model model.pth \
    --output result.jpg \
    --save-crops \
    --crops-dir my_signatures \
    --crop-padding 20 \
    --threshold 0.6

# Process a PDF and extract all signatures
python signature_detector_enhanced.py contract.pdf \
    --model model.pth \
    --output pdf_results \
    --save-crops \
    --no-show \
    --threshold 0.5

# Process with default settings (crops saved to auto-generated directory)
python signature_detector_enhanced.py document.jpg \
    --model model.pth \
    --save-crops
"""


# ============================================
# Example 4: Programmatic crop extraction only
# ============================================
# from PIL import Image

# # Load image and detect
# image = Image.open("document.jpg")
# detections = detector.detect(image)

# # Save crops without visualization
# if detections:
#     crop_paths = detector.crop_and_save_detections(
#         image=image,
#         detections=detections,
#         output_dir="signatures_only",
#         prefix="sig",
#         padding=10
#     )
#     print(f"Extracted {len(crop_paths)} signatures")