import { useState, useCallback } from 'react';
import { 
  AnalysisConfig, 
  AnalysisStatus, 
  AnalysisResults, 
  UploadResponse,
  StatusResponse,
  SignatureDetection,
  ClusterGroup,
  SimilarityScore,
  PageImage
} from '../types';

const API_BASE = 'http://localhost:5000/api';

export const useAPI = () => {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Check API connection
  const checkConnection = useCallback(async (): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE}/status`);
      if (response.ok) {
        const data: StatusResponse = await response.json();
        console.log('✅ Connected to API:', data.service);
        setIsConnected(true);
        setError(null);
        return true;
      } else {
        setIsConnected(false);
        setError('API health check failed');
        return false;
      }
    } catch (err) {
      console.error('❌ API connection failed:', err);
      setIsConnected(false);
      setError('Cannot connect to backend. Please ensure Flask server is running on localhost:5000');
      return false;
    }
  }, []);

  // Upload reference file
  const uploadReference = useCallback(async (file: File): Promise<string | null> => {
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch(`${API_BASE}/upload-reference`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to upload reference file');
      }
      
      const result: UploadResponse = await response.json();
      
      if (result.job_id) {
        console.log('✅ Reference uploaded, job ID:', result.job_id);
        return result.job_id;
      } else {
        throw new Error('No job ID returned from server');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Upload failed';
      console.error('❌ Reference upload failed:', errorMessage);
      setError(errorMessage);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Upload target files
  const uploadTargets = useCallback(async (files: File[], jobId: string): Promise<boolean> => {
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      files.forEach(file => formData.append('files', file));
      formData.append('job_id', jobId);
      
      const response = await fetch(`${API_BASE}/upload-targets`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to upload target files');
      }
      
      const result: UploadResponse = await response.json();
      
      if (result.status === 'uploaded') {
        console.log('✅ Targets uploaded:', result.files_uploaded, 'files');
        return true;
      } else {
        throw new Error('Upload status not confirmed');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Upload failed';
      console.error('❌ Target upload failed:', errorMessage);
      setError(errorMessage);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  // Start analysis
  const startAnalysis = useCallback(async (jobId: string, config: AnalysisConfig): Promise<boolean> => {
    setLoading(true);
    setError(null);
    
    try {
      // Build request to match Flask backend exactly
      const requestBody = {
        job_id: jobId,
        reference_path: `uploads/${jobId}_reference_${config.reference_filename || 'unknown.pdf'}`,
        target_folder: `uploads/${jobId}_targets`,
        config: {
          model_path: config.model_path,
          similarity_threshold: config.similarity_threshold,
          vgg_weight: config.vgg_weight,
          vit_weight: config.vit_weight
        }
      };
      
      console.log('🚀 Starting analysis with config:', requestBody);
      
      const response = await fetch(`${API_BASE}/start-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to start analysis');
      }
      
      const result = await response.json();
      
      if (result.status === 'started') {
        console.log('✅ Analysis started for job:', jobId);
        return true;
      } else {
        throw new Error('Analysis not started');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Analysis failed to start';
      console.error('❌ Analysis start failed:', errorMessage);
      setError(errorMessage);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  // Get analysis status
  const getAnalysisStatus = useCallback(async (jobId: string): Promise<AnalysisStatus | null> => {
    try {
      const response = await fetch(`${API_BASE}/analysis-status/${jobId}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to get analysis status');
      }
      
      const result = await response.json();
      
      // Return status in the expected format
      return {
        job_id: result.job_id || jobId,
        status: result.status,
        progress: result.progress || 0,
        current_step: result.current_step || 'Processing...',
        error: result.error
      };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Status check failed';
      console.error('❌ Status check failed:', errorMessage);
      setError(errorMessage);
      return null;
    }
  }, []);

  // NEW: Get page images separately (optional endpoint)
  const getPageImages = useCallback(async (jobId: string): Promise<PageImage[] | null> => {
    try {
      const response = await fetch(`${API_BASE}/page-images/${jobId}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to get page images');
      }
      
      const result = await response.json();
      console.log('🖼️ Raw page images from backend:', result);
      
      return transformPageImages(result.page_images || []);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Page images fetch failed';
      console.error('❌ Page images fetch failed:', errorMessage);
      setError(errorMessage);
      return null;
    }
  }, []);

  // ENHANCED: Get analysis results with page image support
  const getAnalysisResults = useCallback(async (jobId: string): Promise<AnalysisResults | null> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE}/analysis-results/${jobId}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to get analysis results');
      }
      
      const result = await response.json();
      console.log('📊 Raw backend results:', result);
      
      // ENHANCED: Transform Flask backend response to match frontend expectations
      const transformedResults: AnalysisResults = {
        job_id: jobId,
        // Transform signatures with preserved page_number references
        signatures: transformSignatures(result.signer_profiles || {}, result.page_images || []),
        clusters: transformClusters(result.signer_profiles || {}),
        similarity_matrix: transformSimilarityMatrix(result.similarity_matrix || []),
        // NEW: Transform page images for page view
        page_images: transformPageImages(result.page_images || []),
        // Keep backend-specific fields
        level1_signers: result.level1_signers || 0,
        level2_final_signers: result.level2_final_signers || 0,
        signer_profiles: result.signer_profiles || {},
        target_results: result.target_results || {},
        report_path: result.report_path
      };
      
      console.log('✅ Transformed results with page images:', transformedResults);
      console.log(`📄 Found ${transformedResults.page_images.length} page images`);
      console.log(`🔍 Found ${transformedResults.signatures.length} signatures`);
      
      return transformedResults;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get results';
      console.error('❌ Results fetch failed:', errorMessage);
      setError(errorMessage);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Download report
  const downloadReport = useCallback(async (jobId: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE}/download-report/${jobId}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to download report');
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `signature_crossref_report_${jobId}.xlsx`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      
      console.log('✅ Report downloaded successfully');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Download failed';
      console.error('❌ Download failed:', errorMessage);
      setError(errorMessage);
    }
  }, []);

  // Cleanup job
  const cleanupJob = useCallback(async (jobId: string): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE}/cleanup/${jobId}`, {
        method: 'DELETE'
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to cleanup job');
      }
      
      console.log('✅ Job cleaned up successfully');
      return true;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Cleanup failed';
      console.error('❌ Cleanup failed:', errorMessage);
      setError(errorMessage);
      return false;
    }
  }, []);

  return {
    isConnected,
    loading,
    error,
    checkConnection,
    uploadReference,
    uploadTargets,
    startAnalysis,
    getAnalysisStatus,
    getAnalysisResults,
    getPageImages, // NEW: Dedicated page images endpoint
    downloadReport,
    cleanupJob,
    setError,
  };
};

// NEW: Transform page images from backend to frontend format
function transformPageImages(backendPageImages: any[]): PageImage[] {
  if (!Array.isArray(backendPageImages)) {
    console.warn('⚠️ Page images data is not an array:', backendPageImages);
    return [];
  }

  return backendPageImages.map((pageData: any) => {
    // Handle different possible backend formats
    const pageImage: PageImage = {
      pdf_name: pageData.pdf_name || pageData.document_name || 'Unknown Document',
      page_number: pageData.page_number || pageData.page || 1,
      image_data: pageData.image_data || pageData.base64_image || '',
      dimensions: {
        width: pageData.dimensions?.width || pageData.width || 800,
        height: pageData.dimensions?.height || pageData.height || 1000
      },
      signatures: (pageData.signatures || []).map((sig: any) => ({
        unique_id: sig.unique_id || sig.id || `sig_${Math.random().toString(36).substr(2, 9)}`,
        bounding_box: sig.bounding_box || [0, 0, 100, 50],
        confidence_score: sig.confidence_score || sig.confidence || 0.8,
        bbox_coordinates: sig.bbox_coordinates || {
          x1: sig.bounding_box?.[0] || 0,
          y1: sig.bounding_box?.[1] || 0,
          x2: sig.bounding_box?.[2] || 100,
          y2: sig.bounding_box?.[3] || 50
        }
      }))
    };

    console.log(`📄 Transformed page ${pageImage.page_number} with ${pageImage.signatures.length} signatures`);
    return pageImage;
  });
}

// ENHANCED: Transform signatures with better page_number preservation
function transformSignatures(signerProfiles: Record<string, any>, pageImages: any[] = []): SignatureDetection[] {
  const signatures: SignatureDetection[] = [];
  
  Object.entries(signerProfiles).forEach(([signerId, profile]) => {
    if (profile.signature_details && Array.isArray(profile.signature_details)) {
      profile.signature_details.forEach((sig: any) => {
        // Enhanced signature transformation with page context
        const signature: SignatureDetection = {
          id: sig.unique_id || `${signerId}_${Math.random().toString(36).substr(2, 9)}`,
          bounding_box: sig.bounding_box || { x: 0, y: 0, width: 100, height: 50 },
          confidence_score: sig.confidence_score || 0.8,
          page_number: sig.page_number || 1, // PRESERVED: page_number reference
          image_data: sig.image_data,
          // NEW: Enhanced fields for page context
          pdf_name: sig.pdf_name || profile.pdf_name || 'Unknown',
          page_image_ref: `${sig.pdf_name || 'doc'}_page${sig.page_number || 1}`,
          bbox_coordinates: sig.bbox_coordinates || {
            x1: sig.bounding_box?.x || 0,
            y1: sig.bounding_box?.y || 0,
            x2: (sig.bounding_box?.x || 0) + (sig.bounding_box?.width || 100),
            y2: (sig.bounding_box?.y || 0) + (sig.bounding_box?.height || 50),
            original_box: sig.original_bounding_box || []
          }
        };
        
        signatures.push(signature);
      });
    }
  });
  
  console.log(`🔍 Transformed ${signatures.length} signatures with page references`);
  return signatures;
}

// Transform clusters (unchanged but improved logging)
function transformClusters(signerProfiles: Record<string, any>): ClusterGroup[] {
  const clusters: ClusterGroup[] = [];
  const colors = ['#3B82F6', '#F97316', '#10B981', '#8B5CF6', '#F59E0B', '#EF4444'];
  let colorIndex = 0;
  
  Object.entries(signerProfiles).forEach(([signerId, profile]) => {
    if (profile.signature_details && Array.isArray(profile.signature_details)) {
      // Create individual clusters for each signer
      const cluster: ClusterGroup = {
        id: signerId,
        name: signerId.replace('final_signer_', 'Signer '),
        type: 'individual',
        signatures: profile.signature_details.map((sig: any) => ({
          id: sig.unique_id || `${signerId}_${Math.random().toString(36).substr(2, 9)}`,
          bounding_box: sig.bounding_box || { x: 0, y: 0, width: 100, height: 50 },
          confidence_score: sig.confidence_score || 0.8,
          page_number: sig.page_number || 1, // PRESERVED: page_number
          image_data: sig.image_data,
          pdf_name: sig.pdf_name,
          page_image_ref: `${sig.pdf_name || 'doc'}_page${sig.page_number || 1}`
        })),
        confidence: profile.confidence_score || 0.8,
        color: colors[colorIndex % colors.length]
      };
      
      clusters.push(cluster);
      colorIndex++;
    }
  });
  
  console.log(`👥 Transformed ${clusters.length} clusters`);
  return clusters;
}

// Transform similarity matrix (unchanged)
function transformSimilarityMatrix(similarityData: any[]): SimilarityScore[] {
  // Transform your backend similarity data to the expected format
  if (!Array.isArray(similarityData)) {
    return [];
  }
  
  return similarityData.map((item: any) => ({
    signature1_id: item.signature1_id || item.sig1 || '',
    signature2_id: item.signature2_id || item.sig2 || '',
    similarity: item.similarity || item.score || 0,
    features: {
      hog: item.features?.hog || item.hog || 0,
      resnet50: item.features?.resnet50 || item.resnet50 || 0,
      vgg19: item.features?.vgg19 || item.vgg19 || 0,
      vit: item.features?.vit || item.vit || 0
    }
  }));
}