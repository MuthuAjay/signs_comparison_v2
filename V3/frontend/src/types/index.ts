// NEW: Page Image Interface
export interface PageImage {
  pdf_name: string;
  page_number: number;
  image_data: string; // base64 encoded image data
  dimensions: {
    width: number;
    height: number;
  };
  signatures: SignatureBoundingBox[]; // Array of signature bounding boxes on this page
}

// NEW: Bounding Box Overlay Interface for animated boxes
export interface BoundingBoxOverlay {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence_score: number;
  isSelected?: boolean;
  isHovered?: boolean;
  animationType?: 'pulse' | 'glow' | 'bounce' | 'none';
  color?: string;
  strokeWidth?: number;
}

// NEW: Signature Bounding Box for page overlays
export interface SignatureBoundingBox {
  unique_id: string;
  bounding_box: number[]; // [x1, y1, x2, y2] format from backend
  confidence_score: number;
  bbox_coordinates?: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
}

// UPDATED: SignatureDetection interface with page_number reference
export interface SignatureDetection {
  id: string;
  bounding_box: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidence_score: number;
  page_number: number; // Reference to parent page
  image_data?: string;
  // NEW: Additional fields for page context
  pdf_name?: string;
  page_image_ref?: string; // Reference to the PageImage this signature belongs to
  bbox_coordinates?: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    original_box: number[];
  };
}

export interface ClusterGroup {
  id: string;
  name: string;
  type: 'style' | 'individual';
  signatures: SignatureDetection[];
  confidence: number;
  color: string;
}

export interface SimilarityScore {
  signature1_id: string;
  signature2_id: string;
  similarity: number;
  features: {
    hog: number;
    resnet50: number;
    vgg19: number;
    vit: number;
  };
}

export interface AnalysisConfig {
  similarity_threshold: number;
  model_path: string;
  vgg_weight: number;
  vit_weight: number;
  reference_filename?: string; // Added optional field for tracking
}

export interface AnalysisStatus {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'error';
  progress: number;
  current_step: string;
  error?: string;
}

// UPDATED: AnalysisResults to include page images
export interface AnalysisResults {
  job_id: string;
  signatures: SignatureDetection[];
  clusters: ClusterGroup[];
  similarity_matrix: SimilarityScore[];
  // NEW: Page images for full page display
  page_images: PageImage[];
  // Backend-specific fields
  level1_signers?: number;
  level2_final_signers?: number;
  signer_profiles?: Record<string, any>;
  target_results?: Record<string, any>;
  report_path?: string;
  report_url?: string;
}

export interface APIResponse<T = any> {
  success?: boolean;
  data?: T;
  error?: string;
  job_id?: string;
  status?: string;
  message?: string;
}

// Helper types for demo modes
export type DemoMode = 'full' | 'detection' | 'clustering' | 'simulation';

// NEW: View mode for signature display
export type ViewMode = 'grid' | 'page';

// Upload response types
export interface UploadResponse {
  job_id: string;
  filename?: string;
  filepath?: string;
  status: string;
  files_uploaded?: number;
}

// Status response from backend
export interface StatusResponse {
  status: string;
  service: string;
  version?: string;
  timestamp?: string;
}

// NEW: Page Image Response from API
export interface PageImageResponse {
  job_id: string;
  page_images: PageImage[];
  total_pages: number;
}

// NEW: Signature selection state for cross-view sync
export interface SignatureSelection {
  selectedSignatureIds: string[];
  hoveredSignatureId?: string;
  focusedPageNumber?: number;
}

// NEW: Animation configuration for bounding boxes
export interface AnimationConfig {
  enablePulse: boolean;
  enableHover: boolean;
  enableClick: boolean;
  confidenceColorMapping: boolean;
  animationSpeed: 'slow' | 'medium' | 'fast';
}