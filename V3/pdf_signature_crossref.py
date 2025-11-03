#!/usr/bin/env python3
"""
PDF Signature Cross-Reference System with Two-Level Clustering
Extracts signatures from a reference PDF, clusters them twice (two levels), then searches for those signers in other PDFs
"""

import os
import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
import numpy as np
from collections import defaultdict
import cv2
from sklearn.metrics.pairwise import cosine_similarity

# Import your existing modules
try:
    from processor import FeatureExtractor, PDFSignatureAnalyzer, SignatureDetector
    from signs_clustering import SignatureClustering
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("Make sure processor.py and signs_clustering.py are available")


class SignatureMatcher:
    """Handles signature matching between reference and target signatures"""
    
    def __init__(self, similarity_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        self.feature_extractor = None
    
    def initialize_feature_extractor(self):
        """Initialize feature extractor for signature matching"""
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor()
        return self.feature_extractor
    
    def extract_signature_features(self, signature_path):
        """Extract features from a single signature using your existing feature extractor"""
        try:
            feature_extractor = self.initialize_feature_extractor()
            
            # Load image
            from PIL import Image
            signature_image = Image.open(signature_path)
            
            # Use your existing extract_all_features method
            features_dict = feature_extractor.extract_all_features(signature_image)
            
            # Debug: Check what we got
            print(f"     🔍 Features extracted: {list(features_dict.keys())}")
            for key, value in features_dict.items():
                if value is not None:
                    value_type = type(value)
                    if hasattr(value, 'shape'):
                        print(f"       {key}: {value_type} shape {value.shape}")
                    elif hasattr(value, '__len__'):
                        print(f"       {key}: {value_type} length {len(value)}")
                    else:
                        print(f"       {key}: {value_type}")
                else:
                    print(f"       {key}: None")
            
            # Extract VGG and ViT features (your clustering system uses these)
            vgg_features = features_dict.get('vgg19')
            vit_features = features_dict.get('vit')
            
            # Convert to numpy arrays if they're lists
            if vgg_features is not None:
                vgg_features = np.array(vgg_features) if isinstance(vgg_features, list) else vgg_features
                vgg_features = vgg_features.flatten()
            
            if vit_features is not None:
                vit_features = np.array(vit_features) if isinstance(vit_features, list) else vit_features
                vit_features = vit_features.flatten()
            
            # Use the same feature fusion logic as your clustering system
            combined_features = self.fuse_features(vgg_features, vit_features, 
                                                  vgg_weight=0.6, vit_weight=0.4)
            
            print(f"     ✅ Combined features shape: {combined_features.shape}")
            return combined_features
            
        except Exception as e:
            print(f"     ❌ Error extracting features from {signature_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def fuse_features(self, vgg_features, vit_features, vgg_weight=0.6, vit_weight=0.4):
        """Fuse VGG19 and ViT features with dimension alignment and weighted combination"""
        if vit_features is None or (hasattr(vit_features, '__len__') and len(vit_features) == 0):
            print("     ⚠️ ViT features unavailable, using only VGG19")
            if vgg_features is not None:
                vgg_features = np.array(vgg_features).flatten()
                return vgg_features / np.linalg.norm(vgg_features) if np.linalg.norm(vgg_features) > 0 else vgg_features
            else:
                raise ValueError("No valid features available")
        
        # Handle different feature dimensions
        vgg_features = np.array(vgg_features).flatten()
        vit_features = np.array(vit_features).flatten()
        
        print(f"     🔍 Feature dimensions: VGG19={vgg_features.shape}, ViT={vit_features.shape}")
        
        # Method 1: Resize smaller feature to match larger (padding/truncating)
        if len(vgg_features) > len(vit_features):
            # Pad ViT features to match VGG19 size
            pad_size = len(vgg_features) - len(vit_features)
            vit_features = np.pad(vit_features, (0, pad_size), mode='constant', constant_values=0)
        elif len(vit_features) > len(vgg_features):
            # Truncate ViT features to match VGG19 size
            vit_features = vit_features[:len(vgg_features)]
        
        # Normalize each feature type
        vgg_norm = vgg_features / np.linalg.norm(vgg_features) if np.linalg.norm(vgg_features) > 0 else vgg_features
        vit_norm = vit_features / np.linalg.norm(vit_features) if np.linalg.norm(vit_features) > 0 else vit_features
        
        # Weighted combination
        fused = vgg_weight * vgg_norm + vit_weight * vit_norm
        return fused / np.linalg.norm(fused) if np.linalg.norm(fused) > 0 else fused
    
    def compare_signatures(self, reference_features, target_features):
        """Compare two signature feature vectors"""
        if reference_features is None or target_features is None:
            return 0.0
        
        # Ensure features are numpy arrays
        reference_features = np.array(reference_features) if not isinstance(reference_features, np.ndarray) else reference_features
        target_features = np.array(target_features) if not isinstance(target_features, np.ndarray) else target_features
        
        # Ensure features are 2D arrays for cosine similarity
        if reference_features.ndim == 1:
            reference_features = reference_features.reshape(1, -1)
        if target_features.ndim == 1:
            target_features = target_features.reshape(1, -1)
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(reference_features, target_features)[0][0]
        return float(similarity)
    
    def find_matching_signer(self, target_signature_path, reference_signer_features):
        """Find which reference signer (if any) matches the target signature"""
        target_features = self.extract_signature_features(target_signature_path)
        if target_features is None:
            return None, 0.0
        
        best_match_signer = None
        best_similarity = 0.0
        
        for signer_id, signer_feature_list in reference_signer_features.items():
            # Compare with all signatures of this signer and take the best match
            max_similarity_for_signer = 0.0
            
            for ref_features in signer_feature_list:
                similarity = self.compare_signatures(ref_features, target_features)
                max_similarity_for_signer = max(max_similarity_for_signer, similarity)
            
            if max_similarity_for_signer > best_similarity:
                best_similarity = max_similarity_for_signer
                best_match_signer = signer_id
        
        # Only return match if above threshold
        if best_similarity >= self.similarity_threshold:
            return best_match_signer, best_similarity
        else:
            return None, best_similarity


class PDFSignatureCrossReference:
    """Main class for PDF signature cross-reference analysis with two-level clustering"""
    
    def __init__(self, model_path, output_dir="signature_crossref_output", similarity_threshold=0.7):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.pdf_analyzer = None
        self.signature_matcher = SignatureMatcher(similarity_threshold)
        
        # Results storage for two-level clustering
        self.level1_signers = {}  # First level clustering results
        self.level2_signers = {}  # Second level clustering results (final signers)
        self.reference_signer_features = {}  # Features for final signers
        self.target_pdf_results = {}
        
        self.signature_metadata = {}
        
        # Initialize PDF analyzer
        if model_path and os.path.exists(model_path):
            print(f"🧠 Initializing PDF signature detector...")
            self.pdf_analyzer = PDFSignatureAnalyzer(
                model_path=model_path,
                output_dir=str(self.output_dir / "temp_signatures")
            )
            print(f"✅ PDF analyzer ready")
        else:
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
    def process_reference_pdf(self, reference_pdf_path, clustering_params=None):
        """Process reference PDF and perform two-level clustering"""
        print(f"📄 Processing reference PDF: {reference_pdf_path}")
        
        try:
            # Extract signatures from reference PDF
            ref_results = self.pdf_analyzer.detect_signatures_in_pdf(str(reference_pdf_path))
            
            if not ref_results['signatures_detected']:
                print(f"❌ No signatures found in reference PDF")
                return False
            
            print(f"✅ Found {len(ref_results['signatures_detected'])} signatures in reference PDF")
            
            # CRITICAL FIX: Store original detection metadata for Flask backend access
            for sig_metadata in ref_results['signatures_detected']:
                sig_path = sig_metadata['signature_path']
                self.signature_metadata[sig_path] = sig_metadata
            
            print(f"🔗 Stored metadata for {len(ref_results['signatures_detected'])} signatures")
            
            # Prepare signature data for clustering
            signature_data = []
            for sig_metadata in ref_results['signatures_detected']:
                signature_data.append({
                    'path': sig_metadata['signature_path'],
                    'source_pdf': Path(reference_pdf_path).name,
                    'source_page': sig_metadata['page_number'],
                    'signature_id': sig_metadata['unique_id'],
                    'confidence': sig_metadata['confidence_score'],
                    'metadata': sig_metadata
                })
            
            # LEVEL 1 CLUSTERING
            print(f"🎯 Starting LEVEL 1 clustering...")
            level1_success = self._perform_level1_clustering(signature_data, clustering_params)
            if not level1_success:
                return False
            
            # LEVEL 2 CLUSTERING
            print(f"🎯 Starting LEVEL 2 clustering...")
            level2_success = self._perform_level2_clustering(clustering_params)
            if not level2_success:
                return False
            
            # CRITICAL FIX: Link final signatures to original metadata
            self._link_signatures_to_metadata()
            
            # Extract features for final signers (Level 2 results)
            self._extract_final_signer_features()
            
            print(f"✅ Two-level clustering completed with {len(self.level2_signers)} final signers")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing reference PDF: {e}")
            import traceback
            print(f"Full traceback:")
            traceback.print_exc()
            return False    
            
            
    def _link_signatures_to_metadata(self):
        """Link final clustered signatures back to their original detection metadata"""
        print(f"🔗 Linking final signatures to original detection metadata...")
        
        # Check if we have original metadata stored
        if not hasattr(self, 'signature_metadata') or not self.signature_metadata:
            print(f"⚠️ No original signature metadata found to link")
            return
        
        print(f"📊 Original metadata available for {len(self.signature_metadata)} signatures")
        print(f"📊 Level 2 signers to process: {len(self.level2_signers)}")
        
        # Debug: Show what metadata we have
        print(f"🔍 Sample metadata keys: {list(self.signature_metadata.keys())[:3]}")
        
        linked_count = 0
        not_found_count = 0
        
        # Process each final signer (Level 2 results)
        for signer_id, profile in self.level2_signers.items():
            signature_paths = profile.get('all_signature_paths', [])
            linked_metadata_list = []
            
            print(f"   👤 Processing {signer_id} with {len(signature_paths)} signatures...")
            
            for sig_path in signature_paths:
                if not os.path.exists(sig_path):
                    print(f"     ⚠️ Signature file not found: {sig_path}")
                    not_found_count += 1
                    continue
                
                # Extract the original filename from the path
                sig_filename = os.path.basename(sig_path)
                
                # Try multiple strategies to find the original metadata
                original_metadata = None
                
                # Strategy 1: Direct path match
                if sig_path in self.signature_metadata:
                    original_metadata = self.signature_metadata[sig_path]
                    print(f"     ✅ Direct path match: {sig_filename}")
                
                # Strategy 2: Search by filename pattern
                if original_metadata is None:
                    for metadata_path, metadata in self.signature_metadata.items():
                        metadata_filename = os.path.basename(metadata_path)
                        # Check if filenames match (accounting for potential renaming during clustering)
                        if metadata_filename == sig_filename:
                            original_metadata = metadata
                            print(f"     ✅ Filename match: {sig_filename}")
                            break
                
                # Strategy 3: Search by unique_id in filename
                if original_metadata is None:
                    # Extract potential unique_id from filename (remove extensions and prefixes)
                    potential_id = sig_filename.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                    
                    for metadata_path, metadata in self.signature_metadata.items():
                        if metadata.get('unique_id') and potential_id in metadata['unique_id']:
                            original_metadata = metadata
                            print(f"     ✅ ID-based match: {sig_filename} -> {metadata['unique_id']}")
                            break
                
                # Strategy 4: Search by signature characteristics (if available)
                if original_metadata is None:
                    # Look for metadata with similar characteristics
                    # This could be expanded based on your specific metadata structure
                    for metadata_path, metadata in self.signature_metadata.items():
                        metadata_filename = os.path.basename(metadata_path)
                        # Check for partial matches or similar naming patterns
                        if any(part in sig_filename for part in metadata_filename.split('_')[:2]):
                            original_metadata = metadata
                            print(f"     ✅ Pattern match: {sig_filename} -> {metadata_filename}")
                            break
                
                if original_metadata:
                    linked_metadata_list.append(original_metadata)
                    linked_count += 1
                    print(f"     📋 Linked: {sig_filename} -> {original_metadata.get('unique_id', 'unknown_id')}")
                else:
                    not_found_count += 1
                    print(f"     ❌ No metadata found for: {sig_filename}")
                    
                    # Create default metadata as fallback
                    default_metadata = {
                        'unique_id': sig_filename.replace('.png', '').replace('.jpg', '').replace('.jpeg', ''),
                        'signature_path': sig_path,
                        'confidence_score': 0.8,
                        'page_number': 1,
                        'bounding_box': [0, 0, 100, 50],
                        'detection_timestamp': datetime.now().isoformat(),
                        'metadata_source': 'default_fallback'
                    }
                    linked_metadata_list.append(default_metadata)
                    print(f"     🔄 Created default metadata for: {sig_filename}")
            
            # Store linked metadata in the signer profile
            self.level2_signers[signer_id]['linked_metadata'] = linked_metadata_list
            
            print(f"   📊 {signer_id}: {len(linked_metadata_list)} metadata records linked")
        
        print(f"\n📊 Metadata linking summary:")
        print(f"   ✅ Successfully linked: {linked_count}")
        print(f"   ❌ Not found (used defaults): {not_found_count}")
        print(f"   📁 Total final signers: {len(self.level2_signers)}")
        
        # Store linked metadata in a more accessible format for the Flask backend
        self.final_signer_metadata = {}
        for signer_id, profile in self.level2_signers.items():
            linked_metadata = profile.get('linked_metadata', [])
            self.final_signer_metadata[signer_id] = {
                'metadata_list': linked_metadata,
                'signature_count': len(linked_metadata),
                'confidence_score': profile.get('confidence_score', 0.8),
                'source_level1_signer': profile.get('source_level1_signer', 'Unknown')
            }
        
        print(f"🔗 Metadata linking completed for {len(self.final_signer_metadata)} final signers")
        
        return True        
    def _perform_level1_clustering(self, signature_data, clustering_params):
        """Perform first level clustering on individual signatures"""
        try:
            # Setup Level 1 clustering
            level1_output_dir = self.output_dir / "level1_clustering"
            level1_output_dir.mkdir(exist_ok=True)
            
            if clustering_params is None:
                clustering_params = {'vgg_weight': 0.7, 'vit_weight': 0.3}
            
            print(f"📊 Level 1 clustering with parameters: {clustering_params}")
            
            feature_extractor = FeatureExtractor()
            level1_clustering = SignatureClustering(
                output_dir=str(level1_output_dir),
                **clustering_params
            )
            
            # Get signature paths for clustering
            signature_paths = [item['path'] for item in signature_data]
            print(f"📋 Level 1: Clustering {len(signature_paths)} signature paths")
            
            # Run Level 1 clustering
            level1_results = level1_clustering.cluster_image_list(signature_paths, feature_extractor)
            
            # Store Level 1 results
            self.level1_signers = level1_results.get('signer_profiles', {})
            
            print(f"✅ Level 1 completed: {len(self.level1_signers)} initial signer groups")
            
            # Debug: Print Level 1 signer profiles
            for signer_id, profile in self.level1_signers.items():
                print(f"   👤 Level 1 Signer {signer_id}: {profile.get('signature_count', 'unknown')} signatures")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in Level 1 clustering: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _perform_level2_clustering(self, clustering_params):
        """Perform second level clustering - cluster images within each Level 1 signer folder individually"""
        try:
            # Prepare Level 1 clustered signatures directory
            level1_clustered_dir = self.output_dir / "level1_clustering" / "clustered_signatures"
            
            if not level1_clustered_dir.exists():
                print(f"❌ Level 1 clustered signatures directory not found: {level1_clustered_dir}")
                return False
            
            # Setup Level 2 clustering output
            level2_output_dir = self.output_dir / "level2_clustering"
            level2_final_signers_dir = level2_output_dir / "final_signers"
            level2_output_dir.mkdir(exist_ok=True)
            level2_final_signers_dir.mkdir(exist_ok=True)
            
            print(f"📊 Level 2 clustering: Processing each Level 1 signer individually...")
            
            # Get all Level 1 signer folders
            level1_signer_folders = [f for f in level1_clustered_dir.iterdir() if f.is_dir()]
            
            if not level1_signer_folders:
                print(f"❌ No Level 1 signer folders found in {level1_clustered_dir}")
                return False
            
            print(f"🔍 Found {len(level1_signer_folders)} Level 1 signer folders to process")
            
            feature_extractor = FeatureExtractor()
            final_signer_counter = 1
            self.level2_signers = {}
            
            # Process each Level 1 signer folder individually
            for level1_folder in level1_signer_folders:
                level1_signer_id = level1_folder.name
                print(f"\n🎯 Processing Level 1 Signer: {level1_signer_id}")
                
                # Get all signature images in this Level 1 signer folder
                signature_images = []
                for sig_file in level1_folder.iterdir():
                    if sig_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                        signature_images.append(str(sig_file))
                
                if not signature_images:
                    print(f"   ⚠️ No signature images found in {level1_signer_id}")
                    continue
                
                print(f"   📸 Found {len(signature_images)} signatures in {level1_signer_id}")
                
                if len(signature_images) == 1:
                    # Only one signature - create final signer directly
                    final_signer_id = f"Signer_{final_signer_counter:03d}"
                    final_signer_dir = level2_final_signers_dir / final_signer_id
                    final_signer_dir.mkdir(exist_ok=True)
                    
                    # Copy the single signature to final folder
                    import shutil
                    source_file = signature_images[0]
                    dest_file = final_signer_dir / os.path.basename(source_file)
                    shutil.copy2(source_file, dest_file)
                    
                    self.level2_signers[final_signer_id] = {
                        'signature_count': 1,
                        'confidence_score': 1.0,
                        'all_signature_paths': [str(dest_file)],
                        'source_level1_signer': level1_signer_id
                    }
                    
                    print(f"   ✅ Created {final_signer_id} with 1 signature (single image)")
                    final_signer_counter += 1
                    
                elif len(signature_images) >= 2:
                    # Multiple signatures - perform Level 2 clustering
                    print(f"   🎯 Running Level 2 clustering on {len(signature_images)} signatures...")
                    
                    # Create temporary clustering for this Level 1 signer
                    temp_clustering_dir = level2_output_dir / f"temp_{level1_signer_id}"
                    temp_clustering_dir.mkdir(exist_ok=True)
                    
                    level2_clustering = SignatureClustering(
                        output_dir=str(temp_clustering_dir),
                        **clustering_params
                    )
                    
                    # Run clustering on signatures from this Level 1 signer
                    clustering_results = level2_clustering.cluster_image_list(signature_images, feature_extractor)
                    
                    if clustering_results and 'signer_profiles' in clustering_results:
                        # Process the clustering results
                        temp_clustered_dir = temp_clustering_dir / "clustered_signatures"
                        
                        if temp_clustered_dir.exists():
                            # Move each clustered folder to final signers directory
                            for temp_signer_folder in temp_clustered_dir.iterdir():
                                if temp_signer_folder.is_dir():
                                    # Count signatures in this sub-cluster
                                    sub_signatures = []
                                    for sig_file in temp_signer_folder.iterdir():
                                        if sig_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                                            sub_signatures.append(str(sig_file))
                                    
                                    if sub_signatures:
                                        # Create final signer
                                        final_signer_id = f"Signer_{final_signer_counter:03d}"
                                        final_signer_dir = level2_final_signers_dir / final_signer_id
                                        
                                        # Move the entire folder
                                        import shutil
                                        if final_signer_dir.exists():
                                            shutil.rmtree(final_signer_dir)
                                        shutil.move(str(temp_signer_folder), str(final_signer_dir))
                                        
                                        # Update signature paths to new location
                                        final_signature_paths = []
                                        for sig_file in final_signer_dir.iterdir():
                                            if sig_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                                                final_signature_paths.append(str(sig_file))
                                        
                                        self.level2_signers[final_signer_id] = {
                                            'signature_count': len(final_signature_paths),
                                            'confidence_score': 0.8,  # Good confidence for clustered groups
                                            'all_signature_paths': final_signature_paths,
                                            'source_level1_signer': level1_signer_id
                                        }
                                        
                                        print(f"   ✅ Created {final_signer_id} with {len(final_signature_paths)} signatures")
                                        final_signer_counter += 1
                        else:
                            print(f"   ⚠️ No clustered results found for {level1_signer_id}")
                    else:
                        print(f"   ⚠️ Clustering failed for {level1_signer_id}, creating single signer")
                        # Fallback: create single signer with all signatures
                        final_signer_id = f"Signer_{final_signer_counter:03d}"
                        final_signer_dir = level2_final_signers_dir / final_signer_id
                        final_signer_dir.mkdir(exist_ok=True)
                        
                        # Copy all signatures to final folder
                        import shutil
                        final_signature_paths = []
                        for source_file in signature_images:
                            dest_file = final_signer_dir / os.path.basename(source_file)
                            shutil.copy2(source_file, dest_file)
                            final_signature_paths.append(str(dest_file))
                        
                        self.level2_signers[final_signer_id] = {
                            'signature_count': len(final_signature_paths),
                            'confidence_score': 0.6,  # Lower confidence for unclustered group
                            'all_signature_paths': final_signature_paths,
                            'source_level1_signer': level1_signer_id
                        }
                        
                        print(f"   ✅ Created {final_signer_id} with {len(final_signature_paths)} signatures (fallback)")
                        final_signer_counter += 1
                    
                    # Clean up temporary directory
                    import shutil
                    if temp_clustering_dir.exists():
                        shutil.rmtree(temp_clustering_dir)
            
            print(f"\n✅ Level 2 completed: {len(self.level2_signers)} final signers created")
            
            # Debug: Print final signer summary
            print(f"\n📊 Final Signer Summary:")
            for signer_id, profile in self.level2_signers.items():
                print(f"   👥 {signer_id}: {profile['signature_count']} signatures (from Level 1: {profile['source_level1_signer']})")
                # Show first few paths for verification
                paths = profile['all_signature_paths'][:2]
                for path in paths:
                    print(f"      📄 {os.path.basename(path)}")
                if len(profile['all_signature_paths']) > 2:
                    print(f"      📄 ... and {len(profile['all_signature_paths']) - 2} more")
            
            print(f"\n📁 All final signers organized in: {level2_final_signers_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in Level 2 clustering: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_final_signer_features(self):
        """Extract features for final signers (Level 2 results)"""
        print(f"🔍 Extracting features for final signers...")
        
        # Debug: Check what we have in level2_signers
        print(f"📊 Level 2 signers available: {list(self.level2_signers.keys())}")
        
        for signer_id, profile in self.level2_signers.items():
            signer_features = []
            signature_paths = profile.get('all_signature_paths', [])
            
            print(f"   Processing Final Signer {signer_id} with {len(signature_paths)} signatures...")
            
            valid_paths_found = 0
            for sig_path in signature_paths:
                # Verify the file exists
                if not os.path.exists(sig_path):
                    print(f"     ⚠️ Signature file not found: {sig_path}")
                    continue
                
                valid_paths_found += 1
                    
                try:
                    features = self.signature_matcher.extract_signature_features(sig_path)
                    if features is not None:
                        signer_features.append(features)
                        print(f"     ✅ Features extracted from {os.path.basename(sig_path)}")
                    else:
                        print(f"     ❌ Failed to extract features from {os.path.basename(sig_path)}")
                except Exception as e:
                    print(f"     ❌ Error extracting features from {os.path.basename(sig_path)}: {e}")
            
            print(f"   📊 {signer_id}: {valid_paths_found} valid paths, {len(signer_features)} features extracted")
            
            if signer_features:
                self.reference_signer_features[signer_id] = signer_features
                print(f"   ✅ Final Signer {signer_id}: {len(signer_features)} signature features extracted")
            else:
                print(f"   ⚠️ Final Signer {signer_id}: No features extracted")
        
        print(f"📊 Final feature extraction summary:")
        print(f"   Level 2 signers: {len(self.level2_signers)}")
        print(f"   Signers with features: {len(self.reference_signer_features)}")
        
        if not self.reference_signer_features:
            print(f"🔄 No features extracted, searching in final signers directory...")
            
            # Search in the final signers directory
            level2_final_signers_dir = self.output_dir / "level2_clustering" / "final_signers"
            
            if level2_final_signers_dir.exists():
                print(f"   🔍 Searching in: {level2_final_signers_dir}")
                
                for signer_folder in level2_final_signers_dir.iterdir():
                    if signer_folder.is_dir():
                        signer_id = signer_folder.name
                        signer_features = []
                        
                        print(f"   📂 Processing folder: {signer_id}")
                        
                        # Get all signature files in this signer folder
                        for sig_file in signer_folder.iterdir():
                            if sig_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                                try:
                                    features = self.signature_matcher.extract_signature_features(str(sig_file))
                                    if features is not None:
                                        signer_features.append(features)
                                        print(f"     ✅ Features extracted from {sig_file.name}")
                                except Exception as e:
                                    print(f"     ❌ Error with {sig_file.name}: {e}")
                        
                        if signer_features:
                            self.reference_signer_features[signer_id] = signer_features
                            print(f"   ✅ {signer_id}: {len(signer_features)} features extracted from folder")
                            
                            # Update level2_signers if this signer wasn't already there
                            if signer_id not in self.level2_signers:
                                signature_paths = [str(f) for f in signer_folder.iterdir() 
                                                 if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']]
                                self.level2_signers[signer_id] = {
                                    'signature_count': len(signature_paths),
                                    'confidence_score': 0.8,
                                    'all_signature_paths': signature_paths
                                }
            
            if not self.reference_signer_features:
                raise Exception("No final signer features could be extracted even after searching all directories")
    
    def process_target_pdf_folder(self, target_folder_path):
        """Process folder of target PDFs and find reference signers"""
        target_folder = Path(target_folder_path)
        
        if not target_folder.exists():
            raise FileNotFoundError(f"Target folder not found: {target_folder}")
        
        # Find all PDF files
        pdf_files = list(target_folder.glob("*.pdf")) + list(target_folder.glob("*.PDF"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {target_folder}")
            return
        
        print(f"📁 Processing {len(pdf_files)} PDFs in target folder...")
        
        for pdf_file in pdf_files:
            print(f"\n📄 Processing: {pdf_file.name}")
            self._process_single_target_pdf(pdf_file)
    
    def _process_single_target_pdf(self, pdf_path):
        """Process a single target PDF"""
        try:
            # Extract signatures from target PDF
            target_results = self.pdf_analyzer.detect_signatures_in_pdf(str(pdf_path))
            
            pdf_name = pdf_path.name
            self.target_pdf_results[pdf_name] = {
                'total_signatures': len(target_results['signatures_detected']),
                'signer_matches': {},
                'unmatched_signatures': 0,
                'processing_status': 'success'
            }
            
            if not target_results['signatures_detected']:
                print(f"   ⚠️ No signatures found in {pdf_name}")
                self.target_pdf_results[pdf_name]['processing_status'] = 'no_signatures'
                return
            
            print(f"   🔍 Found {len(target_results['signatures_detected'])} signatures")
            
            # Initialize signer counters and similarity tracking (using final Level 2 signers)
            signer_counts = {signer_id: 0 for signer_id in self.reference_signer_features.keys()}
            signer_similarities = {signer_id: [] for signer_id in self.reference_signer_features.keys()}
            signature_details = []
            unmatched_count = 0
            
            # Process each signature
            for sig_metadata in target_results['signatures_detected']:
                sig_path = sig_metadata['signature_path']
                
                # Find matching reference signer (from Level 2 final signers)
                matching_signer, similarity = self.signature_matcher.find_matching_signer(
                    sig_path, self.reference_signer_features
                )
                
                # Store signature details
                signature_detail = {
                    'signature_id': sig_metadata['unique_id'],
                    'page_number': sig_metadata['page_number'],
                    'matched_signer': matching_signer,
                    'similarity_score': similarity,
                    'similarity_percentage': f"{similarity * 100:.1f}%",
                    'confidence_level': 'High' if similarity > 0.8 else 'Medium' if similarity > 0.6 else 'Low'
                }
                signature_details.append(signature_detail)
                
                if matching_signer:
                    signer_counts[matching_signer] += 1
                    signer_similarities[matching_signer].append(similarity)
                    print(f"      ✅ Signature matched to Final Signer {matching_signer} (similarity: {similarity:.3f})")
                else:
                    unmatched_count += 1
                    print(f"      ❌ No match found (best similarity: {similarity:.3f})")  
                              
            
            # Calculate similarity statistics for each signer
            signer_similarity_stats = {}
            for signer_id, similarities in signer_similarities.items():
                if similarities:
                    avg_sim = sum(similarities) / len(similarities)
                    signer_similarity_stats[signer_id] = {
                        'avg_similarity': avg_sim,
                        'avg_similarity_percentage': f"{avg_sim * 100:.1f}%",
                        'max_similarity': max(similarities),
                        'min_similarity': min(similarities),
                        'match_count': len(similarities)
                    }
                else:
                    signer_similarity_stats[signer_id] = {
                        'avg_similarity': 0.0,
                        'avg_similarity_percentage': "0.0%",
                        'max_similarity': 0.0,
                        'min_similarity': 0.0,
                        'match_count': 0
                    }

            # Store results
            self.target_pdf_results[pdf_name]['signer_matches'] = signer_counts
            self.target_pdf_results[pdf_name]['signer_similarities'] = signer_similarity_stats
            self.target_pdf_results[pdf_name]['signature_details'] = signature_details
            self.target_pdf_results[pdf_name]['unmatched_signatures'] = unmatched_count
            
            # Summary for this PDF
            matched_signers = [signer for signer, count in signer_counts.items() if count > 0]
            print(f"   📊 Summary: {len(matched_signers)} final signers found, {unmatched_count} unmatched")
            
        except Exception as e:
            print(f"   ❌ Error processing {pdf_path.name}: {e}")
            self.target_pdf_results[pdf_path.name] = {
                'processing_status': 'error',
                'error_message': str(e)
            }
    
    def generate_cross_reference_report(self):
        """Generate comprehensive cross-reference report with two-level clustering info"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"signature_crossref_report_{timestamp}.xlsx"
        
        try:
            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                
                # Main cross-reference table
                self._create_main_crossref_table(writer)
                
                # Two-level clustering summary
                self._create_clustering_summary(writer)
                
                # Reference signers summary
                self._create_reference_summary(writer)
                
                # Detailed target PDF analysis
                self._create_detailed_target_analysis(writer)
                
                # Signer presence matrix
                self._create_signer_presence_matrix(writer)
                
                # Signature similarity analysis
                self._create_signature_similarity_analysis(writer)

                # Signer similarity summary
                self._create_signer_similarity_summary(writer)
            
            print(f"📊 Cross-reference report saved: {report_path}")
            return str(report_path)
            
        except Exception as e:
            print(f"⚠️ Error generating report: {e}")
            return None
    
    def _create_clustering_summary(self, writer):
        """Create summary of two-level clustering results"""
        clustering_data = []
        
        # Level 1 summary
        clustering_data.append({
            'Clustering_Level': 'Level 1',
            'Description': 'Initial signature clustering',
            'Number_of_Signers': len(self.level1_signers),
            'Details': f'Clustered individual signatures into {len(self.level1_signers)} initial signer groups'
        })
        
        # Level 2 summary  
        clustering_data.append({
            'Clustering_Level': 'Level 2',
            'Description': 'Final signer clustering',
            'Number_of_Signers': len(self.level2_signers),
            'Details': f'Re-clustered Level 1 signers into {len(self.level2_signers)} final signer groups'
        })
        
        clustering_df = pd.DataFrame(clustering_data)
        clustering_df.to_excel(writer, sheet_name='Clustering_Summary', index=False)
    
    def _create_main_crossref_table(self, writer):
        """Create the main cross-reference table with counts and presence indicators"""
        # Prepare data for main table
        main_data = []
        
        # Get all reference signer IDs sorted for consistent ordering (using Level 2 final signers)
        reference_signers = sorted(list(self.reference_signer_features.keys()))
        
        for pdf_name, results in self.target_pdf_results.items():
            if results['processing_status'] != 'success':
                row = {
                    'PDF_Name': pdf_name,
                    'Total_Signatures': 0,
                    'Status': results['processing_status']
                }
                if 'error_message' in results:
                    row['Error_Details'] = results['error_message']
                
                # Fill signer columns with 0 counts and No presence
                for signer in reference_signers:
                    row[f'{signer}_Count'] = 0
                    row[f'{signer}_Present'] = 'No'
                
                row['Reference_Signers_Found'] = 0
                row['Signer_Details'] = 'Error'
                
            else:
                row = {
                    'PDF_Name': pdf_name,
                    'Total_Signatures': results['total_signatures'],
                }
                
                # Add individual signer counts and presence as separate columns
                # Add individual signer counts, presence, and similarity as separate columns
                matched_signers = []
                for signer in reference_signers:
                    count = results['signer_matches'].get(signer, 0)
                    similarity_stats = results.get('signer_similarities', {}).get(signer, {})
                    avg_similarity_pct = similarity_stats.get('avg_similarity_percentage', '0.0%')
                    
                    row[f'{signer}_Count'] = count
                    row[f'{signer}_Present'] = 'Yes' if count > 0 else 'No'
                    row[f'{signer}_Similarity%'] = avg_similarity_pct if count > 0 else '0.0%'
                    if count > 0:
                        matched_signers.append(signer)
                
                # Add summary columns
                row['Reference_Signers_Found'] = len(matched_signers)
                row['Signer_Details'] = ';'.join(matched_signers) if matched_signers else 'None'
                row['Status'] = 'success'
            
            main_data.append(row)
        
        # Create DataFrame with specific column order
        # Interleave count, presence, and similarity columns for each signer
        column_order = ['PDF_Name', 'Total_Signatures']
        for signer in reference_signers:
            column_order.extend([f'{signer}_Count', f'{signer}_Present', f'{signer}_Similarity%'])
        column_order.extend(['Reference_Signers_Found', 'Signer_Details', 'Status'])
        
        main_df = pd.DataFrame(main_data)
        # Reorder columns to match desired format
        main_df = main_df.reindex(columns=column_order)
        main_df.to_excel(writer, sheet_name='Cross_Reference_Analysis', index=False)
        
        # Also create a simplified presence-only table
        simplified_data = []
        for pdf_name, results in self.target_pdf_results.items():
            if results['processing_status'] == 'success':
                row = {
                    'PDF_Name': pdf_name,
                    'Total_Signatures': results['total_signatures'],
                }
                
                # Add only presence columns (Yes/No format)
                matched_signers = []
                for signer in reference_signers:
                    count = results['signer_matches'].get(signer, 0)
                    row[signer] = 'Yes' if count > 0 else 'No'
                    if count > 0:
                        matched_signers.append(signer)
                
                row['Reference_Signers_Found'] = len(matched_signers)
                row['Status'] = 'success'
                simplified_data.append(row)
        
        # Create simplified DataFrame
        simplified_columns = ['PDF_Name', 'Total_Signatures'] + reference_signers + ['Reference_Signers_Found', 'Status']
        simplified_df = pd.DataFrame(simplified_data)
        simplified_df = simplified_df.reindex(columns=simplified_columns)
        simplified_df.to_excel(writer, sheet_name='Presence_Summary', index=False)
    
    def _create_reference_summary(self, writer):
        """Create reference signers summary (using Level 2 final signers)"""
        ref_data = []
        
        for signer_id, profile in self.level2_signers.items():
            ref_data.append({
                'Final_Signer_ID': signer_id,
                'Total_Signature_Count': profile['signature_count'],
                'Confidence_Score': f"{profile['confidence_score']:.3f}",
                'Features_Extracted': len(self.reference_signer_features.get(signer_id, []))
            })
        
        ref_df = pd.DataFrame(ref_data)
        ref_df.to_excel(writer, sheet_name='Reference_Signers', index=False)
    
    def _create_detailed_target_analysis(self, writer):
        """Create detailed target PDF analysis"""
        detailed_data = []
        
        for pdf_name, results in self.target_pdf_results.items():
            if results['processing_status'] == 'success':
                for signer_id, count in results['signer_matches'].items():
                    detailed_data.append({
                        'PDF_Name': pdf_name,
                        'Reference_Signer': signer_id,
                        'Signature_Count': count,
                        'Present': 'Yes' if count > 0 else 'No'
                    })
        
        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='Detailed_Analysis', index=False)
    
    def _create_signer_presence_matrix(self, writer):
        """Create a matrix showing signer presence across PDFs"""
        # Get all PDFs and signers
        pdf_names = [name for name, results in self.target_pdf_results.items() 
                    if results['processing_status'] == 'success']
        signer_ids = list(self.reference_signer_features.keys())
        
        if not pdf_names or not signer_ids:
            return
        
        # Create matrix
        matrix_data = []
        for pdf_name in pdf_names:
            row = {'PDF_Name': pdf_name}
            results = self.target_pdf_results[pdf_name]
            
            for signer_id in signer_ids:
                count = results['signer_matches'].get(signer_id, 0)
                row[signer_id] = count
            
            matrix_data.append(row)
        
        matrix_df = pd.DataFrame(matrix_data)
        matrix_df.to_excel(writer, sheet_name='Signer_Presence_Matrix', index=False)
            
    def _create_signature_similarity_analysis(self, writer):
        """Create detailed signature similarity analysis"""
        similarity_data = []
        
        for pdf_name, results in self.target_pdf_results.items():
            if results['processing_status'] == 'success':
                signature_details = results.get('signature_details', [])
                for detail in signature_details:
                    similarity_data.append({
                        'PDF_Name': pdf_name,
                        'Signature_ID': detail['signature_id'],
                        'Page_Number': detail['page_number'],
                        'Matched_Signer': detail['matched_signer'] or 'No Match',
                        'Similarity_Score': detail['similarity_score'],
                        'Similarity_Percentage': detail['similarity_percentage'],
                        'Confidence_Level': detail['confidence_level']
                    })
        
        if similarity_data:
            similarity_df = pd.DataFrame(similarity_data)
            similarity_df.to_excel(writer, sheet_name='Signature_Similarity_Analysis', index=False)

    def _create_signer_similarity_summary(self, writer):
        """Create signer-level similarity summary"""
        summary_data = []
        
        for pdf_name, results in self.target_pdf_results.items():
            if results['processing_status'] == 'success':
                signer_similarities = results.get('signer_similarities', {})
                for signer_id, stats in signer_similarities.items():
                    if stats['match_count'] > 0:
                        summary_data.append({
                            'PDF_Name': pdf_name,
                            'Signer_ID': signer_id,
                            'Match_Count': stats['match_count'],
                            'Avg_Similarity': stats['avg_similarity'],
                            'Avg_Similarity_Percentage': stats['avg_similarity_percentage'],
                            'Max_Similarity': stats['max_similarity'],
                            'Min_Similarity': stats['min_similarity'],
                            'Confidence_Level': 'High' if stats['avg_similarity'] > 0.8 else 'Medium' if stats['avg_similarity'] > 0.6 else 'Low'
                        })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Signer_Similarity_Summary', index=False)        
    
    def run_full_analysis(self, reference_pdf_path, target_folder_path, 
                         clustering_params=None, generate_report=True):
        """Run complete cross-reference analysis with two-level clustering"""
        
        print(f"🚀 PDF Signature Cross-Reference Analysis (Two-Level Clustering)")
        print(f"📄 Reference PDF: {reference_pdf_path}")
        print(f"📁 Target folder: {target_folder_path}")
        print("=" * 70)
        
        # Process reference PDF with two-level clustering
        if not self.process_reference_pdf(reference_pdf_path, clustering_params):
            print(f"❌ Failed to process reference PDF")
            return None
        
        # Process target PDFs
        self.process_target_pdf_folder(target_folder_path)
        
        # Generate report
        report_path = None
        if generate_report:
            report_path = self.generate_cross_reference_report()
        
        # Print summary
        self._print_summary()
        
        return {
            'level1_signers': len(self.level1_signers),
            'level2_signers': len(self.level2_signers),
            'target_pdfs_processed': len(self.target_pdf_results),
            'report_path': report_path,
            'output_dir': str(self.output_dir)
        }
    
    def _print_summary(self):
        """Print analysis summary with two-level clustering info"""
        print(f"\n📊 Two-Level Clustering Analysis Summary:")
        print(f"🎯 Level 1 signers identified: {len(self.level1_signers)}")
        print(f"🎯 Level 2 final signers: {len(self.level2_signers)}")
        print(f"📄 Target PDFs processed: {len(self.target_pdf_results)}")
        
        # Count successful processes
        successful = sum(1 for results in self.target_pdf_results.values() 
                        if results['processing_status'] == 'success')
        print(f"✅ Successfully processed: {successful}")
        
        # Show final signer distribution
        signer_distribution = defaultdict(int)
        for results in self.target_pdf_results.values():
            if results['processing_status'] == 'success':
                for signer_id, count in results['signer_matches'].items():
                    if count > 0:
                        signer_distribution[signer_id] += 1
        
        print(f"\n🔍 Final signer distribution across target PDFs:")
        for signer_id, pdf_count in signer_distribution.items():
            print(f"   Final Signer {signer_id}: found in {pdf_count} PDFs")


def analyze_signature_crossref(reference_pdf, target_folder, model_path, 
                              output_dir="signature_crossref_output", 
                              similarity_threshold=0.7, **kwargs):
    """
    Main function to analyze signature cross-references between PDFs with two-level clustering
    
    Args:
        reference_pdf: Path to reference PDF file
        target_folder: Path to folder containing target PDF files
        model_path: Path to signature detection model
        output_dir: Output directory for results
        similarity_threshold: Threshold for signature matching (0-1)
        **kwargs: Additional parameters like clustering_params
    """
    
    # Initialize cross-reference analyzer with two-level clustering
    analyzer = PDFSignatureCrossReference(
        model_path=model_path,
        output_dir=output_dir,
        similarity_threshold=similarity_threshold
    )
    
    # Run analysis
    results = analyzer.run_full_analysis(
        reference_pdf_path=reference_pdf,
        target_folder_path=target_folder,
        clustering_params=kwargs.get('clustering_params'),
        generate_report=kwargs.get('generate_report', True)
    )
    
    return results


def main():
    """Example usage with two-level clustering"""
    
    # Configuration
    MODEL_PATH = "/home/eyhyd/signature_comparison_v2/backend/models/detection/weights/faster_rcnn_signatures.pth"
    REFERENCE_PDF = "/home/eyhyd/signature_comparison_v2/Copied_PDFs/test/test1.pdf"
    TARGET_FOLDER = "/home/eyhyd/signature_comparison_v2/Copied_PDFs/test/sample"

    
    # Check if paths exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    if not os.path.exists(REFERENCE_PDF):
        print(f"❌ Reference PDF not found: {REFERENCE_PDF}")
        return
    
    if not os.path.exists(TARGET_FOLDER):
        print(f"❌ Target folder not found: {TARGET_FOLDER}")
        return
    
    # Run analysis with two-level clustering
    results = analyze_signature_crossref(
        reference_pdf=REFERENCE_PDF,
        target_folder=TARGET_FOLDER,
        model_path=MODEL_PATH,
        output_dir="sample_2",
        similarity_threshold=0.7,
        clustering_params={'vgg_weight': 0.9, 'vit_weight': 0.1}
    )
    
    if results:
        print(f"\n✅ Two-level clustering analysis completed successfully!")
        print(f"📁 Results saved to: {results['output_dir']}")
        print(f"🎯 Level 1 signers: {results['level1_signers']}")
        print(f"🎯 Level 2 final signers: {results['level2_signers']}")
        if results['report_path']:
            print(f"📊 Report: {results['report_path']}")


if __name__ == "__main__":
    main()