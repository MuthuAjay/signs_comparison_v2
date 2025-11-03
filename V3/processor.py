import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import hog
import torchvision.models as cnn_models
import fitz  # PyMuPDF
import uuid
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json
import warnings

warnings.filterwarnings("ignore")


class SignatureDetector:
    """Enhanced Signature Detector with PDF processing capabilities"""

    def __init__(
        self,
        model_path: str,
        device=None,
        num_classes=4,
        confidence_threshold=0.5,
        input_size=(512, 512),
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size

        # Initialize and load model
        self.model = self._load_model(model_path)
        self.class_names = ["background", "signature", "initials", "stamp"]
        self.signature_class_id = 1

        print(f"✅ SignatureDetector initialized on {self.device}")

    def _load_model(self, model_path):
        """Load the Faster R-CNN model"""
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")

        checkpoint = torch.load(model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint["model"])
        model = model.to(self.device)
        model.eval()
        return model

    def detect(self, image):
        """Detect signatures in an image"""
        orig_size = image.size
        image_resized = image.resize((self.input_size[1], self.input_size[0]))

        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image_resized).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            predictions = self.model(image_tensor)

        pred = predictions[0]
        boxes = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()

        # Filter by confidence
        keep_idxs = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[keep_idxs]
        labels = labels[keep_idxs]
        scores = scores[keep_idxs]

        # Scale boxes back to original size
        if len(boxes) > 0:
            scale_x = orig_size[0] / self.input_size[1]
            scale_y = orig_size[1] / self.input_size[0]

            scaled_boxes = boxes.copy()
            scaled_boxes[:, 0] *= scale_x
            scaled_boxes[:, 1] *= scale_y
            scaled_boxes[:, 2] *= scale_x
            scaled_boxes[:, 3] *= scale_y
        else:
            scaled_boxes = boxes

        return scaled_boxes, labels, scores

    def get_signatures(self, boxes, labels, scores):
        """Filter detection results to get only signatures"""
        signatures = []
        for box, label, score in zip(boxes, labels, scores):
            if label == self.signature_class_id:
                signatures.append({"box": box.tolist(), "score": float(score)})
        return signatures


class FeatureExtractor:
    """Feature extraction for signature comparison"""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # CNN models
        self.resnet_model = None
        self.vgg_model = None
        self.cnn_transform = None

        # ViT model
        self.vit_model = None
        self.vit_transform = None
        self.vit_available = False

        self._init_models()

    def _init_models(self):
        """Initialize all feature extraction models"""
        # Initialize CNN models
        self._init_cnn_models()

        # Initialize ViT model
        self._init_vit_model()

    def _init_cnn_models(self):
        """Initialize CNN models for feature extraction"""
        print("🧠 Initializing CNN models...")

        # ResNet50
        self.resnet_model = cnn_models.resnet50(
            weights=cnn_models.ResNet50_Weights.IMAGENET1K_V1
        )
        self.resnet_model.fc = nn.Identity()
        self.resnet_model = self.resnet_model.to(self.device)
        self.resnet_model.eval()
        print("✅ ResNet50 initialized")

        # VGG19
        self.vgg_model = cnn_models.vgg19(
            weights=cnn_models.VGG19_Weights.IMAGENET1K_V1
        )
        self.vgg_model.classifier = nn.Identity()
        self.vgg_model = self.vgg_model.to(self.device)
        self.vgg_model.eval()
        print("✅ VGG19 initialized")

        # Shared transform
        self.cnn_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _init_vit_model(self):
        """Initialize Vision Transformer model"""
        try:
            import timm

            print("🧠 Initializing ViT model...")
            self.vit_model = timm.create_model("vit_large_patch16_224", pretrained=True)
            self.vit_model.head = nn.Identity()
            self.vit_model = self.vit_model.to(self.device)
            self.vit_model.eval()

            data_config = timm.data.resolve_data_config({}, model=self.vit_model)
            self.vit_transform = timm.data.create_transform(**data_config)

            self.vit_available = True
            print("✅ ViT model initialized successfully")

        except ImportError:
            print("⚠️ timm library not found. ViT functionality will be disabled.")
            self.vit_available = False
        except Exception as e:
            print(f"⚠️ Failed to initialize ViT model: {e}")
            self.vit_available = False

    def preprocess_signature(self, img_array, target_height=128, target_width=256):
        """Preprocess signature for feature extraction"""
        if len(img_array.shape) == 3:
            gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img_array

        _, binary_img = cv2.threshold(
            gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        h_orig, w_orig = binary_img.shape
        if h_orig == 0 or w_orig == 0:
            return np.zeros((target_height, target_width), dtype=np.uint8)

        scale = (
            min(target_width / w_orig, target_height / h_orig)
            if w_orig > 0 and h_orig > 0
            else 1
        )
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)

        if new_w == 0 or new_h == 0:
            return np.zeros((target_height, target_width), dtype=np.uint8)

        resized_img = cv2.resize(
            binary_img, (new_w, new_h), interpolation=cv2.INTER_AREA
        )

        delta_w = target_width - new_w
        delta_h = target_height - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        normalized_img = cv2.copyMakeBorder(
            resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )
        return normalized_img

    def extract_hog_features(self, processed_img):
        """Extract HOG features"""
        fd = hog(
            processed_img,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            visualize=False,
            block_norm="L2-Hys",
        )
        return fd

    def extract_resnet_features(self, image):
        """Extract ResNet50 features"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        input_tensor = self.cnn_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.resnet_model(input_tensor)

        return features.cpu().numpy().flatten()

    def extract_vgg_features(self, image):
        """Extract VGG19 features"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        input_tensor = self.cnn_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.vgg_model(input_tensor)

        return features.cpu().numpy().flatten()

    def extract_vit_features(self, image):
        """Extract Vision Transformer features"""
        if not self.vit_available:
            raise RuntimeError("ViT model is not available")

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        input_tensor = self.vit_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.vit_model(input_tensor)

        return features.cpu().numpy().flatten()

    def extract_all_features(self, image):
        """Extract all available features from an image"""
        # Preprocess for HOG
        img_array = np.array(image) if isinstance(image, Image.Image) else image
        preprocessed = self.preprocess_signature(img_array)

        features = {
            "hog": self.extract_hog_features(preprocessed),
            "resnet50": self.extract_resnet_features(image),
            "vgg19": self.extract_vgg_features(image),
        }

        # Add ViT features if available
        if self.vit_available:
            try:
                features["vit"] = self.extract_vit_features(image)
            except Exception as e:
                print(f"⚠️ Failed to extract ViT features: {e}")
                features["vit"] = None
        else:
            features["vit"] = None

        return features


class PDFSignatureAnalyzer:
    """Main class for PDF signature detection and analysis"""

    def __init__(self, model_path: str, output_dir: str = "signature_analysis"):
        self.detector = SignatureDetector(model_path)
        self.feature_extractor = FeatureExtractor()
        self.output_dir = output_dir
        self.signatures_dir = os.path.join(output_dir, "detected_signatures")
        self.page_images_dir = os.path.join(output_dir, "page_images")  # NEW

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.signatures_dir, exist_ok=True)
        os.makedirs(self.page_images_dir, exist_ok=True)  # NEW

        # Initialize tracking
        self.signature_metadata = {}
        self.similarity_results = []

        print(f"✅ PDFSignatureAnalyzer initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📁 Signatures directory: {self.signatures_dir}")
        print(f"📁 Page images directory: {self.page_images_dir}")  # NEW

    def pdf_to_images(self, pdf_path: str) -> List[Tuple[int, Image.Image, str]]:  # Updated return type
        """Convert PDF pages to images and save them"""
        print(f"📄 Converting PDF to images: {pdf_path}")
        
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        doc = fitz.open(pdf_path)
        images = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")

            # Convert to PIL Image
            import io
            img = Image.open(io.BytesIO(img_data))
            
            # NEW: Save page image with systematic naming
            page_image_filename = f"{pdf_name}_page{page_num+1:03d}.png"
            page_image_path = os.path.join(self.page_images_dir, page_image_filename)
            img.save(page_image_path, "PNG", quality=95)
            
            print(f"💾 Saved page {page_num+1} image: {page_image_filename}")
            
            images.append((page_num + 1, img, page_image_path))  # Include page image path

        doc.close()
        print(f"✅ Converted and saved {len(images)} pages from PDF")
        return images

    def detect_signatures_in_pdf(self, pdf_path: str) -> Dict:
        """Detect signatures in all pages of a PDF"""
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"\n🔍 Processing PDF: {pdf_name}")

        images = self.pdf_to_images(pdf_path)  # Now returns page_image_path too
        pdf_results = {
            "pdf_name": pdf_name,
            "pdf_path": pdf_path,
            "total_pages": len(images),
            "signatures_detected": [],
            "total_signatures": 0,
            "page_images": []  # NEW: Track page images
        }

        for page_num, image, page_image_path in images:  # Updated unpacking
            print(f"🔎 Detecting signatures on page {page_num}...")

            # NEW: Store page image info
            page_info = {
                "page_number": page_num,
                "page_image_path": page_image_path,
                "page_dimensions": {
                    "width": image.width,
                    "height": image.height
                }
            }
            pdf_results["page_images"].append(page_info)

            # Detect signatures
            boxes, labels, scores = self.detector.detect(image)
            signatures = self.detector.get_signatures(boxes, labels, scores)

            if signatures:
                print(f"✅ Found {len(signatures)} signature(s) on page {page_num}")

                for idx, sig_data in enumerate(signatures):
                    # Generate unique identifier
                    unique_id = f"{pdf_name}_page{page_num:03d}_sig{idx+1:03d}_{uuid.uuid4().hex[:8]}"

                    # Crop signature
                    box = sig_data["box"]
                    x1, y1, x2, y2 = [int(coord) for coord in box]

                    # Add margin
                    w, h = x2 - x1, y2 - y1
                    margin_x, margin_y = int(w * 0.05), int(h * 0.05)
                    x1 = max(0, x1 - margin_x)
                    y1 = max(0, y1 - margin_y)
                    x2 = min(image.width, x2 + margin_x)
                    y2 = min(image.height, y2 + margin_y)

                    cropped_signature = image.crop((x1, y1, x2, y2))

                    # Save cropped signature
                    signature_path = os.path.join(
                        self.signatures_dir, f"{unique_id}.png"
                    )
                    cropped_signature.save(signature_path)

                    # Store metadata (ENHANCED)
                    signature_metadata = {
                        "unique_id": unique_id,
                        "pdf_name": pdf_name,
                        "pdf_path": pdf_path,
                        "page_number": page_num,
                        "signature_index": idx + 1,
                        "confidence_score": sig_data["score"],
                        "bounding_box": box,
                        "signature_path": signature_path,
                        "detection_timestamp": datetime.now().isoformat(),
                        # NEW: Page image information
                        "page_image_path": page_image_path,
                        "page_dimensions": {
                            "width": image.width,
                            "height": image.height
                        },
                        "bbox_coordinates": {  # More detailed bbox info
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "original_box": box
                        }
                    }

                    self.signature_metadata[signature_path] = signature_metadata
                    pdf_results["signatures_detected"].append(signature_metadata)
            else:
                print(f"❌ No signatures found on page {page_num}")

        pdf_results["total_signatures"] = len(pdf_results["signatures_detected"])
        print(
            f"📊 PDF Summary: {pdf_results['total_signatures']} signatures found across {pdf_results['total_pages']} pages"
        )
        print(f"🖼️ Page images saved to: {self.page_images_dir}")

        return pdf_results

    def calculate_signature_similarities(self, pdf_results: Dict) -> Dict:
        """Calculate similarities between signatures within a PDF using all methods with VGG19 emphasis"""
        print(f"\n📊 Calculating signature similarities for {pdf_results['pdf_name']}")

        signatures = pdf_results["signatures_detected"]

        if len(signatures) < 2:
            print("⚠️ Less than 2 signatures found. No similarity analysis performed.")
            return {
                "pairwise_comparisons": [],
                "similarity_statistics": {
                    "total_comparisons": 0,
                    "methods": ["hog", "resnet50", "vgg19", "vit"],
                    "avg_similarities": {
                        "hog": 0.0,
                        "resnet50": 0.0,
                        "vgg19": 0.0,
                        "vit": 0.0 if self.feature_extractor.vit_available else None,
                    },
                    "high_similarity_pairs": {
                        "hog_80%": 0,
                        "resnet50_80%": 0,
                        "vgg19_80%": 0,
                        "vit_80%": 0 if self.feature_extractor.vit_available else None,
                        "combined_80%": 0,
                    },
                },
            }

        # Extract features for all signatures using all methods
        signature_features = {}
        for sig in signatures:
            print(f"🧠 Extracting all features for {sig['unique_id']}")

            # Load signature image
            image = Image.open(sig["signature_path"])

            # Extract all features
            features = self.feature_extractor.extract_all_features(image)

            signature_features[sig["unique_id"]] = {
                "features": features,
                "metadata": sig,
            }

        # Calculate pairwise similarities using all methods
        signature_ids = list(signature_features.keys())
        pairwise_comparisons = []

        # Similarity accumulators for statistics
        similarities_sum = {
            "hog": 0.0,
            "resnet50": 0.0,
            "vgg19": 0.0,
            "vit": 0.0 if self.feature_extractor.vit_available else None,
        }

        high_similarity_counts = {
            "hog_80%": 0,
            "resnet50_80%": 0,
            "vgg19_80%": 0,
            "vit_80%": 0 if self.feature_extractor.vit_available else None,
            "combined_80%": 0,
        }

        total_comparisons = 0

        # NEW: Define similarity weights with VGG19 emphasis
        similarity_weights = {
            'vgg19': 0.7,      # 50% - best performer for signatures
            'resnet50': 0.15,  # 25% - complementary CNN features
            'hog': 0.15,       # 15% - traditional shape/edge features
            'vit': 0         # 10% - spatial context (when available)
        }
        
        print(f"🎯 Using weighted similarity calculation:")
        print(f"   VGG19: {similarity_weights['vgg19']*100:.0f}% weight")
        print(f"   ResNet50: {similarity_weights['resnet50']*100:.0f}% weight") 
        print(f"   HOG: {similarity_weights['hog']*100:.0f}% weight")
        print(f"   ViT: {similarity_weights['vit']*100:.0f}% weight")

        for i in range(len(signature_ids)):
            for j in range(i + 1, len(signature_ids)):
                sig1_id = signature_ids[i]
                sig2_id = signature_ids[j]

                sig1_data = signature_features[sig1_id]
                sig2_data = signature_features[sig2_id]

                # Calculate similarities for each method
                similarities = {}

                # HOG similarity
                hog_sim = cosine_similarity(
                    sig1_data["features"]["hog"].reshape(1, -1),
                    sig2_data["features"]["hog"].reshape(1, -1),
                )[0][0]
                similarities["hog"] = float(hog_sim)
                similarities_sum["hog"] += hog_sim
                if hog_sim >= 0.8:
                    high_similarity_counts["hog_80%"] += 1

                # ResNet50 similarity
                resnet_sim = cosine_similarity(
                    sig1_data["features"]["resnet50"].reshape(1, -1),
                    sig2_data["features"]["resnet50"].reshape(1, -1),
                )[0][0]
                similarities["resnet50"] = float(resnet_sim)
                similarities_sum["resnet50"] += resnet_sim
                if resnet_sim >= 0.8:
                    high_similarity_counts["resnet50_80%"] += 1

                # VGG19 similarity
                vgg_sim = cosine_similarity(
                    sig1_data["features"]["vgg19"].reshape(1, -1),
                    sig2_data["features"]["vgg19"].reshape(1, -1),
                )[0][0]
                similarities["vgg19"] = float(vgg_sim)
                similarities_sum["vgg19"] += vgg_sim
                if vgg_sim >= 0.8:
                    high_similarity_counts["vgg19_80%"] += 1

                # ViT similarity (if available)
                if (
                    self.feature_extractor.vit_available
                    and sig1_data["features"]["vit"] is not None
                    and sig2_data["features"]["vit"] is not None
                ):
                    vit_sim = cosine_similarity(
                        sig1_data["features"]["vit"].reshape(1, -1),
                        sig2_data["features"]["vit"].reshape(1, -1),
                    )[0][0]
                    similarities["vit"] = float(vit_sim)
                    similarities_sum["vit"] += vit_sim
                    if vit_sim >= 0.8:
                        high_similarity_counts["vit_80%"] += 1
                else:
                    similarities["vit"] = None

                # NEW: Combined similarity (weighted average with VGG19 emphasis)
                weighted_sum = 0.0
                total_weight = 0.0

                for method, sim_value in similarities.items():
                    if sim_value is not None and method in similarity_weights:
                        weight = similarity_weights[method]
                        weighted_sum += sim_value * weight
                        total_weight += weight

                combined_similarity = weighted_sum / total_weight if total_weight > 0 else 0.0
                similarities["combined"] = float(combined_similarity)

                if combined_similarity >= 0.8:
                    high_similarity_counts["combined_80%"] += 1

                comparison_result = {
                    "pdf_name": pdf_results["pdf_name"],
                    "signature1_id": sig1_id,
                    "signature2_id": sig2_id,
                    "signature1_page": sig1_data["metadata"]["page_number"],
                    "signature2_page": sig2_data["metadata"]["page_number"],
                    "similarities": similarities,
                    "analysis_timestamp": datetime.now().isoformat(),
                    # NEW: Add weight information for transparency
                    "similarity_weights_used": similarity_weights.copy()
                }

                pairwise_comparisons.append(comparison_result)
                self.similarity_results.append(comparison_result)
                total_comparisons += 1

                print(f"🎯 Comparison {total_comparisons}: {sig1_id} vs {sig2_id}")
                print(
                    f"   HOG: {similarities['hog']:.4f}, ResNet: {similarities['resnet50']:.4f}, VGG: {similarities['vgg19']:.4f}"
                    + (
                        f", ViT: {similarities['vit']:.4f}"
                        if similarities["vit"] is not None
                        else ""
                    )
                    + f", Weighted Combined: {similarities['combined']:.4f}"
                )

        # Calculate average similarities
        avg_similarities = {}
        for method in similarities_sum:
            if similarities_sum[method] is not None and total_comparisons > 0:
                avg_similarities[method] = similarities_sum[method] / total_comparisons
            else:
                avg_similarities[method] = None

        similarity_statistics = {
            "total_comparisons": total_comparisons,
            "methods": ["hog", "resnet50", "vgg19"]
            + (["vit"] if self.feature_extractor.vit_available else []),
            "avg_similarities": avg_similarities,
            "high_similarity_pairs": high_similarity_counts,
            # NEW: Add weight configuration to statistics
            "similarity_weights_used": similarity_weights,
            "weighting_method": "VGG19_emphasized"
        }

        print(
            f"✅ Completed {total_comparisons} similarity comparisons using weighted methods"
        )
        print(f"🎯 VGG19 weight: {similarity_weights['vgg19']*100:.0f}% (primary method)")

        return {
            "pairwise_comparisons": pairwise_comparisons,
            "similarity_statistics": similarity_statistics,
        }

    def process_multiple_pdfs(self, pdf_paths: List[str]) -> Dict:
        """Process multiple PDFs for signature detection and analysis"""
        print(f"\n🚀 Starting batch processing of {len(pdf_paths)} PDFs")

        all_results = {
            "processing_summary": {
                "total_pdfs": len(pdf_paths),
                "processed_pdfs": 0,
                "total_signatures_detected": 0,
                "total_similarities_calculated": 0,
                "processing_timestamp": datetime.now().isoformat(),
            },
            "pdf_results": [],
            "error_files": [],
        }

        for pdf_path in pdf_paths:
            try:
                print(f"\n{'='*60}")
                print(f"📄 Processing: {os.path.basename(pdf_path)}")
                print(f"{'='*60}")

                # Detect signatures in PDF
                pdf_results = self.detect_signatures_in_pdf(pdf_path)

                # Calculate similarities
                similarity_analysis = self.calculate_signature_similarities(pdf_results)
                pdf_results["similarity_analysis"] = similarity_analysis

                all_results["pdf_results"].append(pdf_results)
                all_results["processing_summary"]["processed_pdfs"] += 1
                all_results["processing_summary"][
                    "total_signatures_detected"
                ] += pdf_results["total_signatures"]
                all_results["processing_summary"][
                    "total_similarities_calculated"
                ] += similarity_analysis["similarity_statistics"]["total_comparisons"]

            except Exception as e:
                print(f"❌ Error processing {pdf_path}: {str(e)}")
                all_results["error_files"].append(
                    {
                        "pdf_path": pdf_path,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        return all_results

    def export_to_excel(self, results: Dict, excel_filename: str = None) -> str:
        """Export results to Excel with each file as a row and comprehensive comparison results"""
        if excel_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_filename = f"signature_analysis_results_{timestamp}.xlsx"

        excel_path = os.path.join(self.output_dir, excel_filename)

        print(f"\n📊 Exporting results to Excel: {excel_path}")

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

            # 1. FILE-BASED SUMMARY SHEET (Each PDF as a row)
            file_summary_data = []
            for pdf_result in results["pdf_results"]:
                similarity_stats = pdf_result["similarity_analysis"][
                    "similarity_statistics"
                ]

                # Get best method
                best_method_data = max(
                    [
                        (k, v)
                        for k, v in similarity_stats["avg_similarities"].items()
                        if v is not None
                    ],
                    key=lambda x: x[1] if x[1] is not None else 0,
                    default=("None", 0.0),
                )

                file_row = {
                    "PDF_Name": pdf_result["pdf_name"],
                    "Total_Pages": pdf_result["total_pages"],
                    "Signatures_Detected": pdf_result["total_signatures"],
                    "Total_Comparisons": similarity_stats["total_comparisons"],
                    # Average Similarities by Method
                    "Avg_HOG_Similarity": f"{similarity_stats['avg_similarities'].get('hog', 0.0):.4f}",
                    "Avg_ResNet50_Similarity": f"{similarity_stats['avg_similarities'].get('resnet50', 0.0):.4f}",
                    "Avg_VGG19_Similarity": f"{similarity_stats['avg_similarities'].get('vgg19', 0.0):.4f}",
                    "Avg_ViT_Similarity": (
                        f"{similarity_stats['avg_similarities'].get('vit', 0.0):.4f}"
                        if similarity_stats["avg_similarities"].get("vit") is not None
                        else "N/A"
                    ),
                    # High Similarity Pairs (80% threshold)
                    "High_Sim_HOG_80%": similarity_stats["high_similarity_pairs"].get(
                        "hog_80%", 0
                    ),
                    "High_Sim_ResNet50_80%": similarity_stats[
                        "high_similarity_pairs"
                    ].get("resnet50_80%", 0),
                    "High_Sim_VGG19_80%": similarity_stats["high_similarity_pairs"].get(
                        "vgg19_80%", 0
                    ),
                    "High_Sim_ViT_80%": (
                        similarity_stats["high_similarity_pairs"].get("vit_80%", 0)
                        if similarity_stats["high_similarity_pairs"].get("vit_80%")
                        is not None
                        else "N/A"
                    ),
                    "High_Sim_Combined_80%": similarity_stats[
                        "high_similarity_pairs"
                    ].get("combined_80%", 0),
                    # Best performing method
                    "Best_Method_by_Avg": best_method_data[0],
                    "Best_Avg_Similarity": f"{best_method_data[1]:.4f}",
                    # Processing info
                    "Processing_Status": "Success",
                    "Error_Message": None,
                }

                file_summary_data.append(file_row)

            # Add error files to summary
            for error_file in results["error_files"]:
                file_row = {
                    "PDF_Name": os.path.basename(error_file["pdf_path"]),
                    "Total_Pages": "N/A",
                    "Signatures_Detected": "N/A",
                    "Total_Comparisons": "N/A",
                    "Avg_HOG_Similarity": "N/A",
                    "Avg_ResNet50_Similarity": "N/A",
                    "Avg_VGG19_Similarity": "N/A",
                    "Avg_ViT_Similarity": "N/A",
                    "High_Sim_HOG_80%": "N/A",
                    "High_Sim_ResNet50_80%": "N/A",
                    "High_Sim_VGG19_80%": "N/A",
                    "High_Sim_ViT_80%": "N/A",
                    "High_Sim_Combined_80%": "N/A",
                    "Best_Method_by_Avg": "N/A",
                    "Best_Avg_Similarity": "N/A",
                    "Processing_Status": "Error",
                    "Error_Message": error_file["error"],
                }
                file_summary_data.append(file_row)

            if file_summary_data:
                file_summary_df = pd.DataFrame(file_summary_data)
                file_summary_df.to_excel(writer, sheet_name="File_Summary", index=False)

            # 2. OVERALL SUMMARY SHEET
            overall_stats = results["processing_summary"]
            all_avg_similarities = {}
            all_high_similarities = {}

            # Calculate overall averages across all files
            methods = ["hog", "resnet50", "vgg19", "vit"]
            for method in methods:
                method_similarities = []
                method_high_count = 0
                total_comparisons = 0

                for pdf_result in results["pdf_results"]:
                    stats = pdf_result["similarity_analysis"]["similarity_statistics"]
                    if stats["avg_similarities"].get(method) is not None:
                        method_similarities.append(
                            stats["avg_similarities"][method]
                            * stats["total_comparisons"]
                        )
                        total_comparisons += stats["total_comparisons"]
                        method_high_count += stats["high_similarity_pairs"].get(
                            f"{method}_80%", 0
                        )

                if total_comparisons > 0:
                    all_avg_similarities[method] = (
                        sum(method_similarities) / total_comparisons
                    )
                else:
                    all_avg_similarities[method] = 0.0

                all_high_similarities[f"{method}_80%"] = method_high_count

            summary_data = {
                "Metric": [
                    "Total PDFs Processed",
                    "Total Signatures Detected",
                    "Total Similarity Comparisons",
                    "Average HOG Similarity (All Files)",
                    "Average ResNet50 Similarity (All Files)",
                    "Average VGG19 Similarity (All Files)",
                    "Average ViT Similarity (All Files)",
                    "High Similarity Pairs HOG (80%+)",
                    "High Similarity Pairs ResNet50 (80%+)",
                    "High Similarity Pairs VGG19 (80%+)",
                    "High Similarity Pairs ViT (80%+)",
                    "Processing Timestamp",
                    "ViT Available",
                ],
                "Value": [
                    overall_stats["processed_pdfs"],
                    overall_stats["total_signatures_detected"],
                    overall_stats["total_similarities_calculated"],
                    f"{all_avg_similarities.get('hog', 0.0):.4f}",
                    f"{all_avg_similarities.get('resnet50', 0.0):.4f}",
                    f"{all_avg_similarities.get('vgg19', 0.0):.4f}",
                    (
                        f"{all_avg_similarities.get('vit', 0.0):.4f}"
                        if self.feature_extractor.vit_available
                        else "N/A"
                    ),
                    all_high_similarities.get("hog_80%", 0),
                    all_high_similarities.get("resnet50_80%", 0),
                    all_high_similarities.get("vgg19_80%", 0),
                    (
                        all_high_similarities.get("vit_80%", 0)
                        if self.feature_extractor.vit_available
                        else "N/A"
                    ),
                    overall_stats["processing_timestamp"],
                    "Yes" if self.feature_extractor.vit_available else "No",
                ],
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Overall_Summary", index=False)

            # 3. DETAILED PAIRWISE COMPARISONS
            detailed_comparisons = []
            for pdf_result in results["pdf_results"]:
                for comparison in pdf_result["similarity_analysis"][
                    "pairwise_comparisons"
                ]:
                    detailed_row = {
                        "PDF_Name": comparison["pdf_name"],
                        "Signature1_ID": comparison["signature1_id"],
                        "Signature2_ID": comparison["signature2_id"],
                        "Signature1_Page": comparison["signature1_page"],
                        "Signature2_Page": comparison["signature2_page"],
                        "HOG_Similarity": f"{comparison['similarities']['hog']:.4f}",
                        "ResNet50_Similarity": f"{comparison['similarities']['resnet50']:.4f}",
                        "VGG19_Similarity": f"{comparison['similarities']['vgg19']:.4f}",
                        "ViT_Similarity": (
                            f"{comparison['similarities']['vit']:.4f}"
                            if comparison["similarities"]["vit"] is not None
                            else "N/A"
                        ),
                        "Combined_Similarity": f"{comparison['similarities']['combined']:.4f}",
                        "High_Similarity_80%": (
                            "Yes"
                            if comparison["similarities"]["combined"] >= 0.8
                            else "No"
                        ),
                        "Very_High_Similarity_90%": (
                            "Yes"
                            if comparison["similarities"]["combined"] >= 0.9
                            else "No"
                        ),
                        "Analysis_Timestamp": comparison["analysis_timestamp"],
                    }
                    detailed_comparisons.append(detailed_row)

            if detailed_comparisons:
                detailed_df = pd.DataFrame(detailed_comparisons)
                detailed_df.to_excel(
                    writer, sheet_name="Detailed_Comparisons", index=False
                )

            # 4. SIGNATURE METADATA SHEET
            if self.signature_metadata:
                metadata_df = pd.DataFrame(list(self.signature_metadata.values()))
                metadata_df.to_excel(
                    writer, sheet_name="Signature_Metadata", index=False
                )

            # 5. METHOD PERFORMANCE COMPARISON
            method_performance = []
            for method in ["hog", "resnet50", "vgg19", "vit"]:
                if method == "vit" and not self.feature_extractor.vit_available:
                    continue

                total_comparisons = 0
                total_similarity = 0.0
                high_similarity_count = 0

                for pdf_result in results["pdf_results"]:
                    stats = pdf_result["similarity_analysis"]["similarity_statistics"]
                    if stats["avg_similarities"].get(method) is not None:
                        total_comparisons += stats["total_comparisons"]
                        total_similarity += (
                            stats["avg_similarities"][method]
                            * stats["total_comparisons"]
                        )
                        high_similarity_count += stats["high_similarity_pairs"].get(
                            f"{method}_80%", 0
                        )

                if total_comparisons > 0:
                    avg_similarity = total_similarity / total_comparisons
                    high_similarity_rate = (
                        high_similarity_count / total_comparisons
                    ) * 100
                else:
                    avg_similarity = 0.0
                    high_similarity_rate = 0.0

                method_performance.append(
                    {
                        "Method": method.upper(),
                        "Total_Comparisons": total_comparisons,
                        "Average_Similarity": f"{avg_similarity:.4f}",
                        "High_Similarity_Pairs_80%": high_similarity_count,
                        "High_Similarity_Rate_%": f"{high_similarity_rate:.2f}%",
                        "Performance_Rank": 0,  # Will be filled after sorting
                    }
                )

            # Rank methods by average similarity
            method_performance.sort(
                key=lambda x: float(x["Average_Similarity"]), reverse=True
            )
            for i, method in enumerate(method_performance):
                method["Performance_Rank"] = i + 1

            if method_performance:
                method_perf_df = pd.DataFrame(method_performance)
                method_perf_df.to_excel(
                    writer, sheet_name="Method_Performance", index=False
                )

            # 6. LEGACY COMPATIBILITY SHEETS
            # Legacy Similarity Analysis
            if self.similarity_results:
                legacy_similarity_results = []
                for result in self.similarity_results:
                    legacy_result = {
                        "pdf_name": result["pdf_name"],
                        "signature1_id": result["signature1_id"],
                        "signature2_id": result["signature2_id"],
                        "signature1_page": result["signature1_page"],
                        "signature2_page": result["signature2_page"],
                        "hog_similarity": result["similarities"]["hog"],
                        "resnet50_similarity": result["similarities"]["resnet50"],
                        "vgg19_similarity": result["similarities"]["vgg19"],
                        "vit_similarity": (
                            result["similarities"]["vit"]
                            if result["similarities"]["vit"] is not None
                            else "N/A"
                        ),
                        "combined_similarity": result["similarities"]["combined"],
                        "similarity_threshold_70": result["similarities"]["combined"]
                        >= 0.7,
                        "similarity_threshold_80": result["similarities"]["combined"]
                        >= 0.8,
                        "similarity_threshold_90": result["similarities"]["combined"]
                        >= 0.9,
                        "analysis_timestamp": result["analysis_timestamp"],
                    }
                    legacy_similarity_results.append(legacy_result)

                similarity_df = pd.DataFrame(legacy_similarity_results)
                similarity_df.to_excel(
                    writer, sheet_name="Similarity_Analysis_Legacy", index=False
                )

            # Legacy PDF Summary
            pdf_summary = []
            for pdf_result in results["pdf_results"]:
                similarity_stats = pdf_result["similarity_analysis"][
                    "similarity_statistics"
                ]
                pdf_summary.append(
                    {
                        "PDF_Name": pdf_result["pdf_name"],
                        "Total_Pages": pdf_result["total_pages"],
                        "Signatures_Detected": pdf_result["total_signatures"],
                        "Similarity_Comparisons": similarity_stats["total_comparisons"],
                        "Combined_High_Similarity_Pairs_80%": similarity_stats[
                            "high_similarity_pairs"
                        ].get("combined_80%", 0),
                        "Best_Method": max(
                            [
                                (k, v)
                                for k, v in similarity_stats["avg_similarities"].items()
                                if v is not None
                            ],
                            key=lambda x: x[1] if x[1] is not None else 0,
                            default=("None", 0.0),
                        )[0],
                    }
                )

            if pdf_summary:
                pdf_summary_df = pd.DataFrame(pdf_summary)
                pdf_summary_df.to_excel(
                    writer, sheet_name="PDF_Summary_Legacy", index=False
                )

            # 7. ERROR ANALYSIS (if any errors occurred)
            if results["error_files"]:
                error_df = pd.DataFrame(results["error_files"])
                error_df.to_excel(writer, sheet_name="Error_Analysis", index=False)

        print(f"✅ Excel report saved successfully: {excel_path}")

        # Print summary statistics
        print(f"\n📊 EXPORT SUMMARY:")
        print(f"   📄 Files processed: {len(file_summary_data)}")
        print(
            f"   🔍 Total signatures: {results['processing_summary']['total_signatures_detected']}"
        )
        print(
            f"   📊 Total comparisons: {results['processing_summary']['total_similarities_calculated']}"
        )
        print(
            f"   🧠 Methods used: HOG, ResNet50, VGG19"
            + (", ViT" if self.feature_extractor.vit_available else "")
        )

        return excel_path

    # Add this method to your PDFSignatureAnalyzer class
    def run_clustering_analysis(self, results):
        """Run clustering analysis on processed PDF results"""
        print(f"\n🎯 Starting clustering analysis for signer identification...")

        # Initialize clustering
        clustering = SignatureClustering(vgg_weight=0.6, vit_weight=0.4)

        # Run integrated clustering
        clustering_results = clustering.integrate_with_analysis_results(
            results["pdf_results"], self.feature_extractor
        )

        # Export clustering results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clustering_excel = os.path.join(
            self.output_dir, f"clustering_analysis_{timestamp}.xlsx"
        )
        clustering.export_clustering_results(clustering_excel, clustering_results)

        # Add clustering summary to main results
        results["clustering_analysis"] = clustering_results
        results["clustering_excel_path"] = clustering_excel

        return results


# Integration addition for your main PDF analyzer script

# Add this import at the top of your main script
from signs_clustering import SignatureClustering


# Modified main function with clustering
def main():
    """Main function with clustering integration"""

    pdf_dir = r"/home/eyhyd/signature_comparison_v2/Copied_PDFs"

    # Configuration
    MODEL_PATH = "/home/eyhyd/signature_comparison_v2/backend/models/detection/weights/faster_rcnn_signatures.pth"
    PDF_PATHS = [
        os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")
    ]

    OUTPUT_DIR = "signature_analysis_output"

    try:
        # Initialize analyzer
        analyzer = PDFSignatureAnalyzer(MODEL_PATH, OUTPUT_DIR)

        # Process PDFs (existing pipeline)
        results = analyzer.process_multiple_pdfs(PDF_PATHS)

        # Export standard analysis
        excel_path = analyzer.export_to_excel(results)

        # NEW: Run clustering analysis
        results = analyzer.run_clustering_analysis(results)

        # Print comprehensive summary
        print(f"\n{'='*60}")
        print(f"🎉 COMPLETE ANALYSIS FINISHED")
        print(f"{'='*60}")
        print(f"📄 PDFs Processed: {results['processing_summary']['processed_pdfs']}")
        print(
            f"🔍 Signatures Detected: {results['processing_summary']['total_signatures_detected']}"
        )
        print(
            f"📊 Similarity Comparisons: {results['processing_summary']['total_similarities_calculated']}"
        )
        print(
            f"👥 Unique Signers Identified: {results['clustering_analysis']['total_signers_identified']}"
        )
        print(f"📁 Output Directory: {OUTPUT_DIR}")
        print(f"📊 Main Analysis Report: {excel_path}")
        print(f"🎯 Clustering Report: {results['clustering_excel_path']}")
        print(f"🖼️  Cropped Signatures: {analyzer.signatures_dir}")

        return results

    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        raise


# Usage Example:
"""
To integrate clustering into your existing pipeline:

1. Save the clustering script as 'signature_clustering.py'
2. Add the import and method to your main PDF analyzer
3. Call the clustering after main analysis
4. Get two Excel files: main analysis + clustering analysis

# Simple integration:
analyzer = PDFSignatureAnalyzer(MODEL_PATH, OUTPUT_DIR)
results = analyzer.process_multiple_pdfs(PDF_PATHS)
results = analyzer.run_clustering_analysis(results)  # Add this line

# Results will include:
# - results['clustering_analysis']['signer_profiles']
# - results['clustering_analysis']['signature_assignments'] 
# - results['clustering_analysis']['total_signers_identified']
# - results['clustering_excel_path']
"""

# def main():
#     """Main function to demonstrate usage"""

#     pdf_dir = r"/home/eyhyd/signature_comparison_v2/Copied_PDFs"

#     # PDF paths
#     PDF_PATHS = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

#     # Configuration
#     MODEL_PATH = "/home/eyhyd/signature_comparison_v2/backend/models/detection/weights/faster_rcnn_signatures.pth"
#     # PDF_PATHS = ["/home/eyhyd/signature_comparison_v2/Celebrity Contract_2.pdf"]

#     OUTPUT_DIR = "signature_analysis_output"

#     try:
#         # Initialize analyzer
#         analyzer = PDFSignatureAnalyzer(MODEL_PATH, OUTPUT_DIR)

#         # Process PDFs
#         results = analyzer.process_multiple_pdfs(PDF_PATHS)

#         # Export to Excel
#         excel_path = analyzer.export_to_excel(results)

#         # Print final summary
#         print(f"\n{'='*60}")
#         print(f"🎉 ANALYSIS COMPLETE")
#         print(f"{'='*60}")
#         print(f"📄 PDFs Processed: {results['processing_summary']['processed_pdfs']}")
#         print(f"🔍 Signatures Detected: {results['processing_summary']['total_signatures_detected']}")
#         print(f"📊 Similarity Comparisons: {results['processing_summary']['total_similarities_calculated']}")
#         print(f"📁 Output Directory: {OUTPUT_DIR}")
#         print(f"📊 Excel Report: {excel_path}")
#         print(f"🖼️  Cropped Signatures: {analyzer.signatures_dir}")

#         return results, excel_path

#     except Exception as e:
#         print(f"❌ Fatal error: {str(e)}")
#         raise

if __name__ == "__main__":
    # Example usage
    print("🚀 PDF Signature Detection and Analysis System")
    print("=" * 60)

    main()
