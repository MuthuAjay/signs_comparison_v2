import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SignatureClustering:
    """Signature clustering for signer identification across documents"""
    
    def __init__(self, vgg_weight=0.6, vit_weight=0.4, output_dir="signature_analysis"):
        self.vgg_weight = vgg_weight
        self.vit_weight = vit_weight
        self.output_dir = output_dir
        self.clusters_dir = os.path.join(output_dir, "clustered_signatures")
        
        # Create cluster directories
        os.makedirs(self.clusters_dir, exist_ok=True)
        
        # Results storage
        self.signer_profiles = {}
        self.signature_assignments = {}
        self.cluster_results = {}
        
        print(f"✅ SignatureClustering initialized (VGG19: {vgg_weight}, ViT: {vit_weight})")
        print(f"📁 Cluster output directory: {self.clusters_dir}")
    
    def fuse_features(self, vgg_features, vit_features):
        """Fuse VGG19 and ViT features with dimension alignment and weighted combination"""
        if vit_features is None:
            print("⚠️ ViT features unavailable, using only VGG19")
            return vgg_features / np.linalg.norm(vgg_features)
        
        # Handle different feature dimensions
        vgg_features = np.array(vgg_features).flatten()
        vit_features = np.array(vit_features).flatten()
        
        print(f"🔍 Feature dimensions: VGG19={vgg_features.shape}, ViT={vit_features.shape}")
        
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
        fused = self.vgg_weight * vgg_norm + self.vit_weight * vit_norm
        return fused / np.linalg.norm(fused) if np.linalg.norm(fused) > 0 else fused
    
    def auto_determine_clusters(self, features, method='adaptive'):
        """Automatically determine optimal number of clusters"""
        if len(features) < 2:
            return [0] * len(features), 1
        
        if len(features) == 2:
            # For 2 samples, decide based on similarity
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity(features[0].reshape(1, -1), features[1].reshape(1, -1))[0][0]
            if sim >= 0.85:  # High similarity threshold
                return [0, 0], 1  # Same cluster
            else:
                return [0, 1], 2  # Different clusters
        
        # For more samples, try different methods and pick best result
        methods_to_try = []
        
        if method in ['hierarchical', 'adaptive']:
            methods_to_try.append('hierarchical')
        if method in ['threshold', 'adaptive']:
            methods_to_try.append('threshold')
        if method in ['dbscan', 'adaptive']:
            methods_to_try.append('dbscan')
        
        best_result = None
        best_score = -1
        
        for method_name in methods_to_try:
            try:
                if method_name == 'hierarchical':
                    labels, n_clusters = self._hierarchical_auto_cluster(features)
                elif method_name == 'threshold':
                    labels, n_clusters = self._threshold_cluster(features)
                elif method_name == 'dbscan':
                    labels, n_clusters = self._dbscan_auto_cluster(features)
                else:
                    continue
                
                # Validate clustering result
                unique_labels = len(set(labels))
                if unique_labels < 2 or unique_labels >= len(features):
                    continue
                
                # Calculate quality score
                from sklearn.metrics.pairwise import cosine_similarity
                score = self._evaluate_clustering_quality(features, labels)
                
                if score > best_score:
                    best_score = score
                    best_result = (labels, n_clusters)
                    
            except Exception as e:
                print(f"⚠️ Method {method_name} failed: {e}")
                continue
        
        # Fallback if all methods fail
        if best_result is None:
            print("⚠️ All clustering methods failed, using simple similarity grouping")
            return self._simple_similarity_grouping(features)
        
        return best_result
    
    def _evaluate_clustering_quality(self, features, labels):
        """Evaluate clustering quality without using silhouette score if problematic"""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(features, labels, metric='cosine')
        except:
            # Alternative quality measure: intra vs inter cluster distances
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(features)
            
            intra_cluster_sim = 0
            inter_cluster_sim = 0
            intra_count = 0
            inter_count = 0
            
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    sim = similarity_matrix[i][j]
                    if labels[i] == labels[j]:
                        intra_cluster_sim += sim
                        intra_count += 1
                    else:
                        inter_cluster_sim += sim
                        inter_count += 1
            
            avg_intra = intra_cluster_sim / max(intra_count, 1)
            avg_inter = inter_cluster_sim / max(inter_count, 1)
            
            # Good clustering: high intra-cluster similarity, low inter-cluster similarity
            return avg_intra - avg_inter
    
    def _simple_similarity_grouping(self, features, threshold=0.8):
        """Simple fallback clustering based on pairwise similarity"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = cosine_similarity(features)
        labels = [-1] * len(features)
        current_cluster = 0
        
        for i in range(len(features)):
            if labels[i] == -1:  # Unassigned
                labels[i] = current_cluster
                # Find all similar signatures
                for j in range(i + 1, len(features)):
                    if labels[j] == -1 and similarity_matrix[i][j] >= threshold:
                        labels[j] = current_cluster
                current_cluster += 1
        
        return labels, current_cluster
    
    def _hierarchical_auto_cluster(self, features):
        """Use hierarchical clustering with automatic cut determination"""
        # Calculate distance matrix
        distances = pdist(features, metric='cosine')
        linkage_matrix = linkage(distances, method='ward')
        
        # Try different numbers of clusters and find optimal
        max_clusters = min(len(features) - 1, 8)  # Max clusters should be n_samples - 1
        min_clusters = min(2, len(features) - 1)  # Ensure we have at least 2 clusters
        
        if max_clusters < min_clusters:
            # Not enough samples for proper clustering
            print(f"⚠️ Too few samples ({len(features)}) for hierarchical clustering")
            return list(range(len(features))), len(features)
        
        best_score = -1
        best_n_clusters = min_clusters
        best_labels = None
        
        for n_clusters in range(min_clusters, max_clusters + 1):
            try:
                labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
                
                # Check if we actually have the expected number of clusters
                unique_labels = len(set(labels))
                if unique_labels < 2 or unique_labels >= len(features):
                    continue
                
                score = silhouette_score(features, labels, metric='cosine')
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
                    best_labels = labels
                    
            except Exception as e:
                print(f"⚠️ Error with {n_clusters} clusters: {e}")
                continue
        
        # If no valid clustering found, use threshold-based approach
        if best_labels is None:
            print("⚠️ Hierarchical clustering failed, falling back to threshold method")
            return self._threshold_cluster(features, threshold=0.8)
        
        print(f"✅ Optimal clustering: {best_n_clusters} clusters (score: {best_score:.3f})")
        return best_labels, best_n_clusters
    
    def _dbscan_auto_cluster(self, features):
        """Use DBSCAN with automatic parameter selection"""
        best_score = -1
        best_labels = None
        best_n_clusters = 0
        
        # Try different eps values
        for eps in [0.1, 0.15, 0.2, 0.25, 0.3]:
            for min_samples in [2, 3]:
                clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                labels = clustering.fit_predict(features)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters > 1:
                    # Filter out noise points for silhouette calculation
                    mask = labels != -1
                    if np.sum(mask) > 1:
                        score = silhouette_score(features[mask], labels[mask], metric='cosine')
                        if score > best_score:
                            best_score = score
                            best_labels = labels
                            best_n_clusters = n_clusters
        
        if best_labels is None:
            # Fallback: each signature is its own cluster
            best_labels = list(range(len(features)))
            best_n_clusters = len(features)
        
        return best_labels, best_n_clusters
    
    def _threshold_cluster(self, features, threshold=0.85):
        """Simple threshold-based clustering"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = cosine_similarity(features)
        
        labels = [-1] * len(features)
        current_cluster = 0
        
        for i in range(len(features)):
            if labels[i] == -1:  # Unassigned
                labels[i] = current_cluster
                # Find all similar signatures
                for j in range(i + 1, len(features)):
                    if similarity_matrix[i][j] >= threshold:
                        labels[j] = current_cluster
                current_cluster += 1
        
        return labels, current_cluster
    
    def within_pdf_clustering(self, pdf_signatures):
        """Cluster signatures within a single PDF"""
        if len(pdf_signatures) < 2:
            return {sig['unique_id']: 0 for sig in pdf_signatures}
        
        # Extract and fuse features
        features = []
        signature_ids = []
        
        for sig in pdf_signatures:
            vgg_features = sig.get('vgg19_features')
            vit_features = sig.get('vit_features')
            
            if vgg_features is not None:
                fused = self.fuse_features(vgg_features, vit_features)
                features.append(fused)
                signature_ids.append(sig['unique_id'])
        
        if len(features) < 2:
            return {sig_id: 0 for sig_id in signature_ids}
        
        features = np.array(features)
        
        # Auto-determine clusters
        labels, n_clusters = self.auto_determine_clusters(features, method='threshold')
        
        # Create assignment mapping
        assignments = {}
        for sig_id, label in zip(signature_ids, labels):
            assignments[sig_id] = f"PDF_{label}"
        
        return assignments
    
    def cross_pdf_clustering(self, all_signatures):
        """Cluster signatures across all PDFs for signer identification"""
        if len(all_signatures) < 2:
            print("⚠️ Less than 2 signatures for cross-PDF clustering")
            return {}
        
        # Extract and fuse features
        features = []
        signature_data = []
        
        for sig in all_signatures:
            vgg_features = sig.get('vgg19_features')
            vit_features = sig.get('vit_features')
            
            if vgg_features is not None:
                try:
                    fused = self.fuse_features(vgg_features, vit_features)
                    features.append(fused)
                    signature_data.append(sig)
                except Exception as e:
                    print(f"⚠️ Failed to fuse features for {sig.get('unique_id', 'unknown')}: {e}")
                    continue
        
        if len(features) < 2:
            print("⚠️ Not enough valid features for cross-PDF clustering")
            return {}
        
        features = np.array(features)
        print(f"🔍 Cross-PDF clustering on {len(features)} signatures...")
        
        # Auto-determine clusters for signer identification with adaptive method
        try:
            labels, n_clusters = self.auto_determine_clusters(features, method='adaptive')
            print(f"✅ Identified {n_clusters} potential signers")
        except Exception as e:
            print(f"❌ Clustering failed: {e}")
            print("🔄 Using fallback: each signature as separate signer")
            labels = list(range(len(features)))
            n_clusters = len(features)
        
        # Create signer profiles
        signer_assignments = {}
        signer_data = {}
        
        for sig, label in zip(signature_data, labels):
            signer_id = f"Signer_{label+1:03d}"
            signer_assignments[sig['unique_id']] = signer_id
            
            if signer_id not in signer_data:
                signer_data[signer_id] = {
                    'signatures': [],
                    'documents': set(),
                    'pages': set()
                }
            
            signer_data[signer_id]['signatures'].append(sig['unique_id'])
            signer_data[signer_id]['documents'].add(sig['pdf_name'])
            signer_data[signer_id]['pages'].add(f"{sig['pdf_name']}_page_{sig['page_number']}")
        
        # Create signer profiles
        for signer_id, data in signer_data.items():
            try:
                confidence = self._calculate_confidence(data['signatures'], features, labels)
            except:
                confidence = 0.5  # Default confidence if calculation fails
                
            self.signer_profiles[signer_id] = {
                'signer_id': signer_id,
                'signature_count': len(data['signatures']),
                'documents_signed': len(data['documents']),
                'document_list': list(data['documents']),
                'signature_ids': data['signatures'],
                'representative_signature': data['signatures'][0],  # First signature as representative
                'confidence_score': confidence
            }
        
        return signer_assignments
    
    def organize_signatures_by_clusters(self, signature_assignments):
        """Organize signature images into cluster folders"""
        print(f"\n📁 Organizing signatures into cluster folders...")
        
        # Create folders for each signer
        signer_folders = {}
        for sig_id, assignment in signature_assignments.items():
            signer_id = assignment['signer_id']
            if signer_id not in signer_folders:
                signer_folder = os.path.join(self.clusters_dir, signer_id)
                os.makedirs(signer_folder, exist_ok=True)
                signer_folders[signer_id] = signer_folder
                print(f"📂 Created folder: {signer_folder}")
        
        # Copy signatures to respective folders
        for sig_id, assignment in signature_assignments.items():
            signer_id = assignment['signer_id']
            source_path = assignment['signature_path']
            
            if os.path.exists(source_path):
                # Create new filename with signer info
                original_filename = os.path.basename(source_path)
                new_filename = f"{signer_id}_{original_filename}"
                destination_path = os.path.join(signer_folders[signer_id], new_filename)
                
                # Copy file
                import shutil
                shutil.copy2(source_path, destination_path)
                
                # Update assignment with new path
                assignment['cluster_path'] = destination_path
        
        print(f"✅ Organized {len(signature_assignments)} signatures into {len(signer_folders)} cluster folders")
        return signer_folders
        """Calculate confidence score for signer assignment"""
        if len(signature_ids) == 1:
            return 0.5  # Low confidence for single signatures
        
        # Find indices of signatures in this cluster
        cluster_indices = []
        for i, sig_id in enumerate(signature_ids):
            cluster_indices.append(i)
        
        if len(cluster_indices) < 2:
            return 0.5
        
        # Calculate intra-cluster similarity
        cluster_features = all_features[cluster_indices]
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = []
        for i in range(len(cluster_features)):
            for j in range(i + 1, len(cluster_features)):
                sim = cosine_similarity(
                    cluster_features[i].reshape(1, -1),
                    cluster_features[j].reshape(1, -1)
                )[0][0]
                similarities.append(sim)
        
    def _calculate_confidence(self, signature_ids, all_features, all_labels):
        """Calculate confidence score for signer assignment"""
        if len(signature_ids) == 1:
            return 0.5  # Low confidence for single signatures
        
        # Find indices of signatures in this cluster
        cluster_indices = []
        for i, sig_id in enumerate(signature_ids):
            cluster_indices.append(i)
        
        if len(cluster_indices) < 2:
            return 0.5
        
        # Calculate intra-cluster similarity
        cluster_features = all_features[cluster_indices]
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = []
        for i in range(len(cluster_features)):
            for j in range(i + 1, len(cluster_features)):
                sim = cosine_similarity(
                    cluster_features[i].reshape(1, -1),
                    cluster_features[j].reshape(1, -1)
                )[0][0]
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.5
    
    def cluster_image_list(self, image_paths, feature_extractor):
        """Cluster a list of signature images directly"""
        print(f"\n🎯 Clustering {len(image_paths)} signature images...")
        
        all_signatures = []
        
        # Process each image
        for i, image_path in enumerate(image_paths):
            if not os.path.exists(image_path):
                print(f"⚠️ Image not found: {image_path}")
                continue
            
            try:
                # Load image and extract features
                from PIL import Image
                signature_image = Image.open(image_path)
                features = feature_extractor.extract_all_features(signature_image)
                
                # Create signature data
                filename = os.path.basename(image_path)
                unique_id = f"img_{i+1:03d}_{filename.split('.')[0]}"
                
                sig_data = {
                    'unique_id': unique_id,
                    'image_path': image_path,
                    'pdf_name': 'DirectInput',
                    'page_number': i + 1,
                    'vgg19_features': features['vgg19'],
                    'vit_features': features['vit'],
                    'signature_path': image_path
                }
                
                all_signatures.append(sig_data)
                print(f"✅ Processed: {filename}")
                
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                continue
        
        if len(all_signatures) < 2:
            print("⚠️ Need at least 2 valid signatures for clustering")
            return {}
        
        # Run cross-image clustering
        cross_image_assignments = self.cross_pdf_clustering(all_signatures)
        
        # Create assignments
        for sig in all_signatures:
            sig_id = sig['unique_id']
            
            self.signature_assignments[sig_id] = {
                'signature_id': sig_id,
                'image_path': sig['image_path'],
                'pdf_name': sig['pdf_name'],
                'page_number': sig['page_number'],
                'within_pdf_cluster': 'N/A',
                'signer_id': cross_image_assignments.get(sig_id, 'Unknown'),
                'signature_path': sig['signature_path']
            }
        
        # Organize into folders
        signer_folders = self.organize_signatures_by_clusters(self.signature_assignments)
        
        return {
            'signer_profiles': self.signer_profiles,
            'signature_assignments': self.signature_assignments,
            'signer_folders': signer_folders,
            'total_signers_identified': len(self.signer_profiles)
        }
    
    def process_clustering(self, similarity_results):
        """Main clustering process using existing similarity analysis results"""
        print(f"\n🎯 Starting signature clustering for signer identification...")
        
        # Extract signature data with features
        all_signatures = []
        pdf_signature_groups = {}
        
        for result in similarity_results:
            # Extract features from the analysis results
            sig1_data = {
                'unique_id': result['signature1_id'],
                'pdf_name': result['pdf_name'],
                'page_number': result['signature1_page'],
                'vgg19_features': None,  # Will be populated from metadata
                'vit_features': None
            }
            
            sig2_data = {
                'unique_id': result['signature2_id'],
                'pdf_name': result['pdf_name'],
                'page_number': result['signature2_page'],
                'vgg19_features': None,
                'vit_features': None
            }
            
            # Group by PDF
            pdf_name = result['pdf_name']
            if pdf_name not in pdf_signature_groups:
                pdf_signature_groups[pdf_name] = []
            
            # Add unique signatures
            if sig1_data not in pdf_signature_groups[pdf_name]:
                pdf_signature_groups[pdf_name].append(sig1_data)
                all_signatures.append(sig1_data)
            
            if sig2_data not in pdf_signature_groups[pdf_name]:
                pdf_signature_groups[pdf_name].append(sig2_data)
                all_signatures.append(sig2_data)
        
        return self._cluster_with_feature_extraction(all_signatures, pdf_signature_groups)
    
    def _cluster_with_feature_extraction(self, all_signatures, pdf_signature_groups):
        """Cluster signatures using feature extraction results"""
        # This method would need to be integrated with the main pipeline
        # to access the actual extracted features
        print("⚠️ Feature extraction integration needed for full clustering")
        return {}
    
    def integrate_with_analysis_results(self, pdf_results_list, feature_extractor):
        """Integrate clustering with existing PDF analysis results"""
        print(f"\n🎯 Starting integrated signature clustering...")
        
        all_signatures = []
        pdf_signature_groups = {}
        
        # Extract signature data with features from PDF results
        for pdf_result in pdf_results_list:
            pdf_name = pdf_result['pdf_name']
            pdf_signature_groups[pdf_name] = []
            
            for sig_metadata in pdf_result['signatures_detected']:
                # Load signature image and extract features
                from PIL import Image
                signature_image = Image.open(sig_metadata['signature_path'])
                features = feature_extractor.extract_all_features(signature_image)
                
                sig_data = {
                    'unique_id': sig_metadata['unique_id'],
                    'pdf_name': pdf_name,
                    'page_number': sig_metadata['page_number'],
                    'vgg19_features': features['vgg19'],
                    'vit_features': features['vit'],
                    'signature_path': sig_metadata['signature_path']
                }
                
                pdf_signature_groups[pdf_name].append(sig_data)
                all_signatures.append(sig_data)
        
        print(f"📊 Processing {len(all_signatures)} signatures across {len(pdf_signature_groups)} PDFs")
        
        # Step 1: Within-PDF clustering
        within_pdf_results = {}
        for pdf_name, signatures in pdf_signature_groups.items():
            print(f"🔍 Within-PDF clustering for {pdf_name} ({len(signatures)} signatures)")
            within_pdf_results[pdf_name] = self.within_pdf_clustering(signatures)
        
        # Step 2: Cross-PDF clustering for signer identification
        cross_pdf_assignments = self.cross_pdf_clustering(all_signatures)
        
        # Step 3: Combine results
        for sig in all_signatures:
            sig_id = sig['unique_id']
            pdf_name = sig['pdf_name']
            
            self.signature_assignments[sig_id] = {
                'signature_id': sig_id,
                'pdf_name': pdf_name,
                'page_number': sig['page_number'],
                'within_pdf_cluster': within_pdf_results.get(pdf_name, {}).get(sig_id, 'Unknown'),
                'signer_id': cross_pdf_assignments.get(sig_id, 'Unknown'),
                'signature_path': sig['signature_path']
            }
        
        # Step 3: Organize signatures into cluster folders
        signer_folders = self.organize_signatures_by_clusters(self.signature_assignments)
        
        return {
            'signer_profiles': self.signer_profiles,
            'signature_assignments': self.signature_assignments,
            'within_pdf_clusters': within_pdf_results,
            'signer_folders': signer_folders,
            'total_signers_identified': len(self.signer_profiles)
        }
    
    def export_clustering_results(self, output_path, results):
        """Export clustering results to Excel"""
        print(f"\n📊 Exporting clustering results to: {output_path}")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # Sheet 1: Signer Identification
            if self.signer_profiles:
                signer_data = []
                for signer_id, profile in self.signer_profiles.items():
                    signer_data.append({
                        'Signer_ID': profile['signer_id'],
                        'Signature_Count': profile['signature_count'],
                        'Documents_Signed': profile['documents_signed'],
                        'Document_List': ', '.join(profile['document_list']),
                        'Representative_Signature': profile['representative_signature'],
                        'Confidence_Score': f"{profile['confidence_score']:.3f}",
                        'Confidence_Level': 'High' if profile['confidence_score'] > 0.8 else 'Medium' if profile['confidence_score'] > 0.6 else 'Low'
                    })
                
                signer_df = pd.DataFrame(signer_data)
                signer_df.to_excel(writer, sheet_name='Signer_Identification', index=False)
            
            # Sheet 2: Signature-Signer Mapping
            if self.signature_assignments:
                mapping_data = []
                for sig_id, assignment in self.signature_assignments.items():
                    mapping_data.append({
                        'Signature_ID': assignment['signature_id'],
                        'PDF_Name': assignment['pdf_name'],
                        'Page_Number': assignment['page_number'],
                        'Signer_ID': assignment['signer_id'],
                        'Within_PDF_Cluster': assignment['within_pdf_cluster'],
                        'Signature_Path': assignment['signature_path']
                    })
                
                mapping_df = pd.DataFrame(mapping_data)
                mapping_df.to_excel(writer, sheet_name='Signature_Signer_Mapping', index=False)
            
            # Sheet 3: Cross-Document Analysis
            if self.signer_profiles:
                cross_doc_data = []
                signers_by_doc = {}
                
                # Group signers by document
                for assignment in self.signature_assignments.values():
                    pdf_name = assignment['pdf_name']
                    signer_id = assignment['signer_id']
                    
                    if pdf_name not in signers_by_doc:
                        signers_by_doc[pdf_name] = set()
                    signers_by_doc[pdf_name].add(signer_id)
                
                # Find document pairs with common signers
                pdf_names = list(signers_by_doc.keys())
                for i in range(len(pdf_names)):
                    for j in range(i + 1, len(pdf_names)):
                        pdf1, pdf2 = pdf_names[i], pdf_names[j]
                        common_signers = signers_by_doc[pdf1] & signers_by_doc[pdf2]
                        
                        if common_signers:
                            cross_doc_data.append({
                                'Document_Pair': f"{pdf1} - {pdf2}",
                                'Common_Signers': len(common_signers),
                                'Shared_Signer_IDs': ', '.join(sorted(common_signers)),
                                'Relationship_Strength': 'High' if len(common_signers) > 2 else 'Medium' if len(common_signers) > 1 else 'Low'
                            })
                
                if cross_doc_data:
                    cross_doc_df = pd.DataFrame(cross_doc_data)
                    cross_doc_df.to_excel(writer, sheet_name='Cross_Document_Analysis', index=False)
            
            # Sheet 4: Clustering Summary
            summary_data = [{
                'Total_Signatures_Processed': len(self.signature_assignments),
                'Unique_Signers_Identified': len(self.signer_profiles),
                'Average_Signatures_Per_Signer': len(self.signature_assignments) / max(len(self.signer_profiles), 1),
                'High_Confidence_Signers': sum(1 for p in self.signer_profiles.values() if p['confidence_score'] > 0.8),
                'Single_Signature_Signers': sum(1 for p in self.signer_profiles.values() if p['signature_count'] == 1),
                'Multi_Document_Signers': sum(1 for p in self.signer_profiles.values() if p['documents_signed'] > 1),
                'Processing_Timestamp': datetime.now().isoformat()
            }]
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Clustering_Summary', index=False)
        
        print(f"✅ Clustering results exported successfully")
        print(f"📈 Identified {len(self.signer_profiles)} unique signers from {len(self.signature_assignments)} signatures")

def main():
    """Example usage for both PDF pipeline and direct image clustering"""
    
    print("🚀 Signature Clustering Pipeline")
    print("=" * 50)
    
    # Example 1: Direct image clustering
    print("\n📋 Example 1: Direct Image Clustering")
    
    # List of signature images
    image_paths = [
        "signature1.png",
        "signature2.png", 
        "signature3.png"
        # Add your image paths here
    ]
    
    # Note: You would need a feature extractor instance
    # from your main pipeline for this to work
    print("⚠️ Need FeatureExtractor instance for direct clustering")
    
    # Example usage:
    # clustering = SignatureClustering(vgg_weight=0.6, vit_weight=0.4)
    # results = clustering.cluster_image_list(image_paths, feature_extractor)
    # clustering.export_clustering_results("image_clustering_results.xlsx", results)
    
    print("\n📋 Example 2: Pipeline Integration")
    print("✅ Ready for integration with existing PDF pipeline")

def cluster_images_standalone(image_paths, output_dir="clustering_output"):
    """Standalone function to cluster a list of signature images"""
    
    # This function would need the feature extractor from your main pipeline
    # For now, it's a template showing how to use it
    
    print(f"🎯 Standalone clustering of {len(image_paths)} images")
    print(f"📁 Output directory: {output_dir}")
    
    # You would call this with:
    # from your_main_script import FeatureExtractor
    # feature_extractor = FeatureExtractor()
    # clustering = SignatureClustering(output_dir=output_dir)
    # results = clustering.cluster_image_list(image_paths, feature_extractor)
    # clustering.export_clustering_results(f"{output_dir}/clustering_results.xlsx", results)
    
    return "Template function - needs feature extractor integration"

if __name__ == "__main__":
    main()