#!/usr/bin/env python3
"""
Flask Backend for PDF Signature Cross-Reference System
Integrates with the two-level clustering Python script
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import uuid
from datetime import datetime
import json
import threading
from werkzeug.utils import secure_filename
import base64
from PIL import Image
import io

# Import your existing PDF analysis script
from pdf_signature_crossref import PDFSignatureCrossReference, analyze_signature_crossref
from processor import PDFSignatureAnalyzer  # Add this import

app = Flask(__name__)
CORS(app)  # Enable CORS for web app communication

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'pdf'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Store analysis jobs in memory (in production, use Redis/database)
analysis_jobs = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_signature_image(image_path):
    """Convert signature image to base64 data URL for frontend display"""
    try:
        if not os.path.exists(image_path):
            return None
            
        with Image.open(image_path) as img:
            # Resize if too large (optional)
            max_size = (200, 100)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

@app.route('/api/status', methods=['GET'])
def status():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'service': 'PDF Signature Cross-Reference API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/upload-reference', methods=['POST'])
def upload_reference():
    """Upload reference PDF file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        job_id = str(uuid.uuid4())
        filepath = os.path.join(UPLOAD_FOLDER, f"{job_id}_reference_{filename}")
        file.save(filepath)
        
        return jsonify({
            'job_id': job_id,
            'filename': filename,
            'filepath': filepath,
            'status': 'uploaded'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/upload-targets', methods=['POST'])
def upload_targets():
    """Upload target PDF files"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    job_id = request.form.get('job_id')
    
    if not job_id:
        return jsonify({'error': 'Job ID required'}), 400
    
    uploaded_files = []
    target_folder = os.path.join(UPLOAD_FOLDER, f"{job_id}_targets")
    os.makedirs(target_folder, exist_ok=True)
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(target_folder, filename)
            file.save(filepath)
            uploaded_files.append({
                'filename': filename,
                'filepath': filepath
            })
    
    return jsonify({
        'job_id': job_id,
        'target_folder': target_folder,
        'files_uploaded': len(uploaded_files),
        'files': uploaded_files,
        'status': 'uploaded'
    })

@app.route('/api/start-analysis', methods=['POST'])
def start_analysis():
    """Start the two-level clustering analysis"""
    data = request.get_json()
    
    job_id = data.get('job_id')
    reference_path = data.get('reference_path').replace(" ", "_")
    target_folder = data.get('target_folder')
    
    print(reference_path)
    
    # Configuration parameters
    config = data.get('config', {})
    model_path = config.get('model_path', '/home/eyhyd/signature_comparison_v2/backend/models/detection/weights/faster_rcnn_signatures.pth')
    similarity_threshold = float(config.get('similarity_threshold', 0.7))
    vgg_weight = float(config.get('vgg_weight', 0.7))
    vit_weight = float(config.get('vit_weight', 0.3))
    
    # Validate inputs
    if not all([job_id, reference_path, target_folder]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    if not os.path.exists(reference_path):
        return jsonify({'error': 'Reference PDF not found'}), 404
    
    if not os.path.exists(target_folder):
        return jsonify({'error': 'Target folder not found'}), 404
    
    # Initialize job tracking
    analysis_jobs[job_id] = {
        'status': 'starting',
        'progress': 0,
        'current_step': 'Initializing analysis...',
        'start_time': datetime.now().isoformat(),
        'results': None,
        'error': None
    }
    
    # Start analysis in background thread
    thread = threading.Thread(
        target=run_analysis_background,
        args=(job_id, reference_path, target_folder, model_path, similarity_threshold, vgg_weight, vit_weight)
    )
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'started',
        'message': 'Analysis started in background'
    })

def encode_page_image(image_path):
    """Convert page image to base64 data URL for frontend display"""
    try:
        if not os.path.exists(image_path):
            return None
            
        with Image.open(image_path) as img:
            # For page images, we might want a larger thumbnail
            max_size = (800, 1000)  # Larger than signature thumbnails
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error encoding page image {image_path}: {e}")
        return None

def run_analysis_background(job_id, reference_path, target_folder, model_path, similarity_threshold, vgg_weight, vit_weight):
    """Run the analysis in background with progress updates"""
    try:
        # Update progress tracking
        def update_progress(progress, step):
            if job_id in analysis_jobs:
                analysis_jobs[job_id]['progress'] = progress
                analysis_jobs[job_id]['current_step'] = step
        
        # Step 1: Initialize (5%)
        update_progress(5, 'Initializing PDF signature detector...')
        
        output_dir = os.path.join(RESULTS_FOLDER, job_id)
        
        # Create analyzer with two-level clustering
        analyzer = PDFSignatureCrossReference(
            model_path=model_path,
            output_dir=output_dir,
            similarity_threshold=similarity_threshold
        )
        
        # Step 2: Process reference PDF (25%)
        update_progress(25, 'Processing reference PDF with two-level clustering...')
        
        clustering_params = {
            'vgg_weight': vgg_weight,
            'vit_weight': vit_weight
        }
        
        success = analyzer.process_reference_pdf(reference_path, clustering_params)
        if not success:
            raise Exception("Failed to process reference PDF")
        
        # NEW: Calculate similarity matrix for reference signatures
        update_progress(35, 'Calculating signature similarities...')
        similarity_matrix = []

        if hasattr(analyzer, 'level2_signers') and len(analyzer.level2_signers) > 1:
            # Get all reference signatures for similarity calculation
            all_ref_signatures = []
            for signer_id, profile in analyzer.level2_signers.items():
                signature_paths = profile.get('all_signature_paths', [])
                for sig_path in signature_paths:
                    if os.path.exists(sig_path):
                        all_ref_signatures.append({
                            'unique_id': os.path.basename(sig_path).replace('.png', ''),
                            'signature_path': sig_path,
                            'page_number': 1,  # Add required page_number
                            'signer_id': signer_id
                        })
            
            # Calculate similarities using existing processor
            if len(all_ref_signatures) > 1:
                pdf_analyzer = PDFSignatureAnalyzer(
                    model_path=model_path,
                    output_dir=output_dir
                )
                
                # Format data structure to match what calculate_signature_similarities expects
                pdf_results_for_similarity = {
                    'pdf_name': 'reference_comparison',
                    'signatures_detected': all_ref_signatures  # Use correct key name
                }
                
                similarity_analysis = pdf_analyzer.calculate_signature_similarities(pdf_results_for_similarity)
                
                # Transform to expected format
                for comparison in similarity_analysis.get('pairwise_comparisons', []):
                    similarity_matrix.append({
                        'signature1_id': comparison['signature1_id'],
                        'signature2_id': comparison['signature2_id'],
                        'similarity': comparison['similarities']['combined'],
                        'features': {
                            'hog': comparison['similarities']['hog'],
                            'resnet50': comparison['similarities']['resnet50'],
                            'vgg19': comparison['similarities']['vgg19'],
                            'vit': comparison['similarities']['vit']
                        }
                    })
        
        # Step 3: Process target PDFs (70%)
        update_progress(70, 'Processing target PDFs and matching signatures...')
        
        analyzer.process_target_pdf_folder(target_folder)
        
        # Step 4: Generate report (90%)
        update_progress(90, 'Generating cross-reference report...')
        
        report_path = analyzer.generate_cross_reference_report()
        
        # Step 5: Process page images and enhance results (95%)
        update_progress(95, 'Processing page images for visualization...')
        
        # NEW: Build page images data structure
        page_images = []
        
        # Access the PDF analyzer's page images from the underlying PDFSignatureAnalyzer
        if hasattr(analyzer, 'pdf_analyzer') and hasattr(analyzer.pdf_analyzer, 'signature_metadata'):
            # Group signatures by PDF and page
            pdf_page_groups = {}
            
            for sig_path, metadata in analyzer.pdf_analyzer.signature_metadata.items():
                pdf_name = metadata['pdf_name']
                page_num = metadata['page_number']
                
                key = f"{pdf_name}_page_{page_num}"
                if key not in pdf_page_groups:
                    pdf_page_groups[key] = {
                        'pdf_name': pdf_name,
                        'page_number': page_num,
                        'page_image_path': metadata.get('page_image_path'),
                        'page_dimensions': metadata.get('page_dimensions', {}),
                        'signatures': []
                    }
                
                # Add signature bbox info to this page
                pdf_page_groups[key]['signatures'].append({
                    'unique_id': metadata['unique_id'],
                    'bounding_box': metadata['bounding_box'],
                    'confidence_score': metadata['confidence_score'],
                    'bbox_coordinates': metadata.get('bbox_coordinates', {})
                })
            
            # Convert to page images array with base64 data
            for page_key, page_data in pdf_page_groups.items():
                page_image_path = page_data['page_image_path']
                
                if page_image_path and os.path.exists(page_image_path):
                    page_entry = {
                        'pdf_name': page_data['pdf_name'],
                        'page_number': page_data['page_number'],
                        'image_data': encode_page_image(page_image_path),
                        'dimensions': page_data['page_dimensions'],
                        'signatures': page_data['signatures']
                    }
                    page_images.append(page_entry)
        
        # Enhanced results with signature details for frontend visualization
        enhanced_signer_profiles = {}
        
        # Process each signer to include signature details with images
        for signer_id, profile in analyzer.level2_signers.items():
            signature_details = []
            
            # Get all signature paths for this signer
            signature_paths = profile.get('all_signature_paths', [])
            
            for sig_path in signature_paths:
                if os.path.exists(sig_path):
                    # Find original detection metadata for this signature
                    sig_filename = os.path.basename(sig_path)
                    original_metadata = None
                    
                    # Search through analyzer's signature metadata
                    if hasattr(analyzer, 'signature_metadata'):
                        original_metadata = None
                        for metadata_path, metadata in analyzer.signature_metadata.items():
                            if sig_filename in os.path.basename(metadata_path):
                                original_metadata = metadata
                                break
                    
                    # Also check PDF analyzer metadata if available
                    elif hasattr(analyzer, 'pdf_analyzer') and hasattr(analyzer.pdf_analyzer, 'signature_metadata'):
                        for metadata_path, metadata in analyzer.pdf_analyzer.signature_metadata.items():
                            if sig_filename in os.path.basename(metadata_path) or metadata['unique_id'] in sig_filename:
                                original_metadata = metadata
                                break
                                            
                    # Create signature detail with image data
                    signature_detail = {
                        'unique_id': sig_filename.replace('.png', ''),
                        'signature_path': sig_path,
                        'image_data': encode_signature_image(sig_path),
                        'confidence_score': original_metadata.get('confidence_score', 0.8) if original_metadata else 0.8,
                        'page_number': original_metadata.get('page_number', 1) if original_metadata else 1,
                        'pdf_name': original_metadata.get('pdf_name', 'Unknown') if original_metadata else 'Unknown',
                        'bounding_box': {
                            'x': int(original_metadata['bounding_box'][0]) if original_metadata and 'bounding_box' in original_metadata else 0,
                            'y': int(original_metadata['bounding_box'][1]) if original_metadata and 'bounding_box' in original_metadata else 0,
                            'width': int(original_metadata['bounding_box'][2] - original_metadata['bounding_box'][0]) if original_metadata and 'bounding_box' in original_metadata else 100,
                            'height': int(original_metadata['bounding_box'][3] - original_metadata['bounding_box'][1]) if original_metadata and 'bounding_box' in original_metadata else 50
                        }
                    }
                    
                    signature_details.append(signature_detail)
            
            # Enhanced signer profile
            enhanced_signer_profiles[signer_id] = {
                'signature_count': len(signature_details),
                'confidence_score': profile.get('confidence_score', 0.8),
                'all_signature_paths': signature_paths,
                'signature_details': signature_details,
                'source_level1_signer': profile.get('source_level1_signer', 'Unknown')
            }
        
        # Step 6: Complete (100%)
        update_progress(100, 'Analysis complete!')
        
        # Prepare results - ENHANCED with page images
        results = {
            'level1_signers': len(analyzer.level1_signers),
            'level2_final_signers': len(analyzer.level2_signers),
            'target_pdfs_processed': len(analyzer.target_pdf_results),
            'report_path': report_path,
            'output_dir': output_dir,
            'signer_profiles': enhanced_signer_profiles,
            'target_results': analyzer.target_pdf_results,
            'page_images': page_images,
            'similarity_matrix': similarity_matrix,  # ADD THIS LINE
            'completion_time': datetime.now().isoformat()
        }
        
        # Update job status
        analysis_jobs[job_id].update({
            'status': 'completed',
            'progress': 100,
            'current_step': 'Analysis completed successfully!',
            'results': results,
            'end_time': datetime.now().isoformat()
        })
        
    except Exception as e:
        # Handle errors
        analysis_jobs[job_id].update({
            'status': 'failed',
            'error': str(e),
            'end_time': datetime.now().isoformat()
        })
        
@app.route('/api/page-images/<job_id>', methods=['GET'])
def get_page_images(job_id):
    """Get page images with signature bounding boxes"""
    if job_id not in analysis_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = analysis_jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({'error': 'Analysis not completed'}), 400
    
    page_images = job['results'].get('page_images', [])
    
    return jsonify({
        'job_id': job_id,
        'page_images': page_images,
        'total_pages': len(page_images)
    })

@app.route('/api/analysis-status/<job_id>', methods=['GET'])
def get_analysis_status(job_id):
    """Get analysis progress and status"""
    if job_id not in analysis_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = analysis_jobs[job_id]
    return jsonify(job)

@app.route('/api/analysis-results/<job_id>', methods=['GET'])
def get_analysis_results(job_id):
    """Get complete analysis results"""
    if job_id not in analysis_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = analysis_jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({'error': 'Analysis not completed'}), 400
    
    return jsonify(job['results'])

@app.route('/api/download-report/<job_id>', methods=['GET'])
def download_report(job_id):
    """Download the Excel report"""
    if job_id not in analysis_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = analysis_jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({'error': 'Analysis not completed'}), 400
    
    report_path = job['results']['report_path']
    
    if not os.path.exists(report_path):
        return jsonify({'error': 'Report file not found'}), 404
    
    return send_file(
        report_path,
        as_attachment=True,
        download_name=f"signature_crossref_report_{job_id}.xlsx",
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

@app.route('/api/cleanup/<job_id>', methods=['DELETE'])
def cleanup_job(job_id):
    """Clean up job files and data"""
    if job_id in analysis_jobs:
        # Remove uploaded files
        reference_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(f"{job_id}_reference")]
        for f in reference_files:
            try:
                os.remove(os.path.join(UPLOAD_FOLDER, f))
            except:
                pass
        
        # Remove target folder
        target_folder = os.path.join(UPLOAD_FOLDER, f"{job_id}_targets")
        if os.path.exists(target_folder):
            import shutil
            shutil.rmtree(target_folder)
        
        # Remove results folder
        results_folder = os.path.join(RESULTS_FOLDER, job_id)
        if os.path.exists(results_folder):
            import shutil
            shutil.rmtree(results_folder)
        
        # Remove job from memory
        del analysis_jobs[job_id]
        
        return jsonify({'message': 'Job cleaned up successfully'})
    
    return jsonify({'error': 'Job not found'}), 404

if __name__ == '__main__':
    print("🚀 Starting PDF Signature Cross-Reference API Server")
    print("📄 Endpoints:")
    print("   POST /api/upload-reference - Upload reference PDF")
    print("   POST /api/upload-targets - Upload target PDFs") 
    print("   POST /api/start-analysis - Start two-level analysis")
    print("   GET  /api/analysis-status/<job_id> - Check progress")
    print("   GET  /api/analysis-results/<job_id> - Get results")
    print("   GET  /api/download-report/<job_id> - Download Excel report")
    print("   DELETE /api/cleanup/<job_id> - Clean up job data")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)