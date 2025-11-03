import React, { useState, useEffect, useCallback } from 'react';
import { Play, RotateCcw, Keyboard, Monitor, Download } from 'lucide-react';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import AnalysisConfiguration from './components/AnalysisConfig';
import ProgressTracker from './components/ProgressTracker';
import SignatureDetection from './components/SignatureDetection';
import ClusteringAnimation from './components/ClusteringAnimation';
import SimilarityMatrix from './components/SimilarityMatrix';
import ResultsDashboard from './components/ResultsDashboard';
import CrossReferenceResults from './components/CrossReferenceResults';
import { useAPI } from './hooks/useAPI';
import { 
  AnalysisConfig, 
  AnalysisStatus, 
  AnalysisResults,
  SignatureDetection as SignatureDetectionType,
  ClusterGroup,
  SimilarityScore,
  PageImage,
  DemoMode 
} from './types';

function App() {
  // API and connection state
  const {
    isConnected,
    loading,
    error,
    checkConnection,
    uploadReference,
    uploadTargets,
    startAnalysis,
    getAnalysisStatus,
    getAnalysisResults,
    downloadReport,
    cleanupJob,
    setError
  } = useAPI();

  // File state
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [targetFiles, setTargetFiles] = useState<File[]>([]);

  // Analysis state
  const [analysisConfig, setAnalysisConfig] = useState<AnalysisConfig>({
    similarity_threshold: 0.7,
    model_path: 'C:\\Users\\Ajay\\Documents\\GitHub\\signs_comparison\\models\\faster_rcnn_signatures.pth',
    vgg_weight: 0.7,
    vit_weight: 0.3,
    reference_filename: ''
  });

  const [jobId, setJobId] = useState<string | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState<AnalysisStatus | null>(null);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResults | null>(null);

  // UI state
  const [demoMode, setDemoMode] = useState<DemoMode>('full');
  const [analysisInProgress, setAnalysisInProgress] = useState(false);
  const [showKeyboardShortcuts, setShowKeyboardShortcuts] = useState(false);

  // ENHANCED: Mock data with page images for simulation mode
  const mockSignatures: SignatureDetectionType[] = [
    {
      id: 'sig_1',
      bounding_box: { x: 100, y: 200, width: 150, height: 50 },
      confidence_score: 0.95,
      page_number: 1,
      pdf_name: 'Sample_Contract_001',
      page_image_ref: 'Sample_Contract_001_page1',
      bbox_coordinates: {
        x1: 100, y1: 200, x2: 250, y2: 250,
        original_box: [100, 200, 250, 250]
      }
    },
    {
      id: 'sig_2',
      bounding_box: { x: 200, y: 300, width: 120, height: 45 },
      confidence_score: 0.87,
      page_number: 1,
      pdf_name: 'Sample_Contract_001',
      page_image_ref: 'Sample_Contract_001_page1',
      bbox_coordinates: {
        x1: 200, y1: 300, x2: 320, y2: 345,
        original_box: [200, 300, 320, 345]
      }
    },
    {
      id: 'sig_3',
      bounding_box: { x: 150, y: 400, width: 180, height: 60 },
      confidence_score: 0.78,
      page_number: 2,
      pdf_name: 'Sample_Contract_002',
      page_image_ref: 'Sample_Contract_002_page2',
      bbox_coordinates: {
        x1: 150, y1: 400, x2: 330, y2: 460,
        original_box: [150, 400, 330, 460]
      }
    }
  ];

  // NEW: Mock page images for simulation mode
  const mockPageImages: PageImage[] = [
    {
      pdf_name: 'Sample_Contract_001',
      page_number: 1,
      image_data: 'data:image/svg+xml;base64,' + btoa(`
        <svg width="800" height="1000" xmlns="http://www.w3.org/2000/svg">
          <!-- Mock document page -->
          <rect width="800" height="1000" fill="#ffffff" stroke="#e5e7eb" stroke-width="2"/>
          
          <!-- Header -->
          <rect x="50" y="50" width="700" height="80" fill="#f8fafc" stroke="#cbd5e1"/>
          <text x="400" y="100" text-anchor="middle" font-family="Arial" font-size="24" font-weight="bold" fill="#1e293b">
            SIGNATURE ANALYSIS DEMO
          </text>
          
          <!-- Document content lines -->
          ${Array.from({length: 20}, (_, i) => 
            `<rect x="80" y="${180 + i * 25}" width="${Math.random() * 500 + 200}" height="3" fill="#cbd5e1"/>`
          ).join('')}
          
          <!-- Signature areas -->
          <rect x="100" y="200" width="150" height="50" fill="rgba(59, 130, 246, 0.1)" stroke="#3b82f6" stroke-width="2" stroke-dasharray="5,5"/>
          <text x="175" y="230" text-anchor="middle" font-family="Arial" font-size="12" fill="#3b82f6">Signature 1</text>
          
          <rect x="200" y="300" width="120" height="45" fill="rgba(16, 185, 129, 0.1)" stroke="#10b981" stroke-width="2" stroke-dasharray="5,5"/>
          <text x="260" y="327" text-anchor="middle" font-family="Arial" font-size="12" fill="#10b981">Signature 2</text>
          
          <!-- Footer -->
          <rect x="50" y="920" width="700" height="30" fill="#f1f5f9" stroke="#cbd5e1"/>
          <text x="400" y="940" text-anchor="middle" font-family="Arial" font-size="12" fill="#64748b">
            Page 1 - Sample Contract 001
          </text>
        </svg>
      `),
      dimensions: { width: 800, height: 1000 },
      signatures: [
        {
          unique_id: 'sig_1',
          bounding_box: [100, 200, 250, 250],
          confidence_score: 0.95,
          bbox_coordinates: { x1: 100, y1: 200, x2: 250, y2: 250 }
        },
        {
          unique_id: 'sig_2',
          bounding_box: [200, 300, 320, 345],
          confidence_score: 0.87,
          bbox_coordinates: { x1: 200, y1: 300, x2: 320, y2: 345 }
        }
      ]
    },
    {
      pdf_name: 'Sample_Contract_002',
      page_number: 2,
      image_data: 'data:image/svg+xml;base64,' + btoa(`
        <svg width="800" height="1000" xmlns="http://www.w3.org/2000/svg">
          <!-- Mock document page 2 -->
          <rect width="800" height="1000" fill="#ffffff" stroke="#e5e7eb" stroke-width="2"/>
          
          <!-- Header -->
          <rect x="50" y="50" width="700" height="80" fill="#f8fafc" stroke="#cbd5e1"/>
          <text x="400" y="100" text-anchor="middle" font-family="Arial" font-size="24" font-weight="bold" fill="#1e293b">
            CONTRACT CONTINUATION
          </text>
          
          <!-- Document content lines -->
          ${Array.from({length: 18}, (_, i) => 
            `<rect x="80" y="${180 + i * 25}" width="${Math.random() * 500 + 200}" height="3" fill="#cbd5e1"/>`
          ).join('')}
          
          <!-- Signature area -->
          <rect x="150" y="400" width="180" height="60" fill="rgba(245, 158, 11, 0.1)" stroke="#f59e0b" stroke-width="2" stroke-dasharray="5,5"/>
          <text x="240" y="435" text-anchor="middle" font-family="Arial" font-size="12" fill="#f59e0b">Signature 3</text>
          
          <!-- Footer -->
          <rect x="50" y="920" width="700" height="30" fill="#f1f5f9" stroke="#cbd5e1"/>
          <text x="400" y="940" text-anchor="middle" font-family="Arial" font-size="12" fill="#64748b">
            Page 2 - Sample Contract 002
          </text>
        </svg>
      `),
      dimensions: { width: 800, height: 1000 },
      signatures: [
        {
          unique_id: 'sig_3',
          bounding_box: [150, 400, 330, 460],
          confidence_score: 0.78,
          bbox_coordinates: { x1: 150, y1: 400, x2: 330, y2: 460 }
        }
      ]
    }
  ];

  const mockClusters: ClusterGroup[] = [
    {
      id: 'cluster_1',
      name: 'Neat Writers',
      type: 'style',
      signatures: [mockSignatures[0]],
      confidence: 0.92,
      color: '#3B82F6'
    },
    {
      id: 'cluster_2',
      name: 'Messy Writers',
      type: 'style',
      signatures: [mockSignatures[1], mockSignatures[2]],
      confidence: 0.85,
      color: '#F97316'
    },
    {
      id: 'signer_1',
      name: 'Signer A',
      type: 'individual',
      signatures: [mockSignatures[0]],
      confidence: 0.95,
      color: '#10B981'
    },
    {
      id: 'signer_2',
      name: 'Signer B',
      type: 'individual',
      signatures: [mockSignatures[1], mockSignatures[2]],
      confidence: 0.82,
      color: '#8B5CF6'
    }
  ];

  const mockSimilarityMatrix: SimilarityScore[] = [
    {
      signature1_id: 'sig_1',
      signature2_id: 'sig_2',
      similarity: 0.65,
      features: { hog: 0.7, resnet50: 0.6, vgg19: 0.65, vit: 0.65 }
    },
    {
      signature1_id: 'sig_1',
      signature2_id: 'sig_3',
      similarity: 0.45,
      features: { hog: 0.5, resnet50: 0.4, vgg19: 0.45, vit: 0.45 }
    },
    {
      signature1_id: 'sig_2',
      signature2_id: 'sig_3',
      similarity: 0.89,
      features: { hog: 0.9, resnet50: 0.88, vgg19: 0.89, vit: 0.89 }
    }
  ];

  // NEW: Mock target results for cross-reference demo
  const mockTargetResults = {
    'target_document_1.pdf': {
      total_signatures: 3,
      processing_status: 'success',
      signer_matches: {
        'signer_1': 2,
        'signer_2': 1
      },
      signer_similarities: {
        'signer_1': {
          avg_similarity: 0.85,
          avg_similarity_percentage: '85.0%',
          match_count: 2
        },
        'signer_2': {
          avg_similarity: 0.72,
          avg_similarity_percentage: '72.0%',
          match_count: 1
        }
      },
      signature_details: [
        {
          signature_id: 'target_sig_1',
          page_number: 1,
          matched_signer: 'signer_1',
          similarity_score: 0.87,
          similarity_percentage: '87.0%',
          confidence_level: 'High'
        },
        {
          signature_id: 'target_sig_2',
          page_number: 1,
          matched_signer: 'signer_1',
          similarity_score: 0.83,
          similarity_percentage: '83.0%',
          confidence_level: 'High'
        },
        {
          signature_id: 'target_sig_3',
          page_number: 2,
          matched_signer: 'signer_2',
          similarity_score: 0.72,
          similarity_percentage: '72.0%',
          confidence_level: 'Medium'
        }
      ]
    },
    'target_document_2.pdf': {
      total_signatures: 2,
      processing_status: 'success',
      signer_matches: {
        'signer_1': 0,
        'signer_2': 2
      },
      signer_similarities: {
        'signer_1': {
          avg_similarity: 0.0,
          avg_similarity_percentage: '0.0%',
          match_count: 0
        },
        'signer_2': {
          avg_similarity: 0.78,
          avg_similarity_percentage: '78.0%',
          match_count: 2
        }
      },
      signature_details: [
        {
          signature_id: 'target_sig_4',
          page_number: 1,
          matched_signer: 'signer_2',
          similarity_score: 0.76,
          similarity_percentage: '76.0%',
          confidence_level: 'Medium'
        },
        {
          signature_id: 'target_sig_5',
          page_number: 1,
          matched_signer: 'signer_2',
          similarity_score: 0.80,
          similarity_percentage: '80.0%',
          confidence_level: 'High'
        }
      ]
    }
  };

  const mockSignerProfiles = {
    'signer_1': {
      confidence_score: 0.95,
      signature_count: 1,
      cluster_id: 'cluster_1'
    },
    'signer_2': {
      confidence_score: 0.82,
      signature_count: 2,
      cluster_id: 'cluster_2'
    }
  };

  // Define functions first before using them in effects
  const canStartAnalysis = useCallback((): boolean => {
    if (demoMode === 'simulation') return true;
    return referenceFile !== null && targetFiles.length > 0 && isConnected && !loading && !analysisInProgress;
  }, [demoMode, referenceFile, targetFiles, isConnected, loading, analysisInProgress]);

  // Reference file handling with filename tracking
  const handleReferenceUpload = useCallback(async (file: File) => {
    if (!file) {
      setReferenceFile(null);
      return;
    }

    setReferenceFile(file);
    setError(null);

    // Update config with filename for backend
    setAnalysisConfig(prev => ({
      ...prev,
      reference_filename: file.name
    }));

    if (isConnected) {
      try {
        const newJobId = await uploadReference(file);
        if (newJobId) {
          setJobId(newJobId);
          console.log('✅ Job created:', newJobId);
        }
      } catch (err) {
        console.error('❌ Reference upload failed:', err);
      }
    }
  }, [isConnected, uploadReference, setError]);

  // Target files handling
  const handleTargetUpload = useCallback(async (files: File[]) => {
    setTargetFiles(files);
    
    if (files.length > 0 && jobId && isConnected) {
      try {
        const success = await uploadTargets(files, jobId);
        if (success) {
          console.log('✅ Target files uploaded successfully');
        }
      } catch (err) {
        console.error('❌ Target upload failed:', err);
      }
    }
  }, [jobId, isConnected, uploadTargets]);

  // ENHANCED: Analysis start with proper config and mock page images
  const handleStartAnalysis = useCallback(async () => {
    if (demoMode === 'simulation') {
      // Simulate analysis with mock data including page images
      setAnalysisInProgress(true);
      setAnalysisStatus({
        job_id: 'mock_job',
        status: 'processing',
        progress: 0,
        current_step: 'Initializing analysis...'
      });

      // Simulate progress
      const steps = [
        { step: 'Extracting signatures from PDFs...', progress: 20 },
        { step: 'Processing page images...', progress: 35 }, // NEW step
        { step: 'Level 1 Clustering (Style Groups)...', progress: 50 },
        { step: 'Level 2 Clustering (Individuals)...', progress: 65 },
        { step: 'Computing similarity matrix...', progress: 80 },
        { step: 'Generating report...', progress: 95 },
        { step: 'Analysis complete!', progress: 100 }
      ];

      for (const { step, progress } of steps) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        setAnalysisStatus({
          job_id: 'mock_job',
          status: progress === 100 ? 'completed' : 'processing',
          progress,
          current_step: step
        });
      }

      setAnalysisInProgress(false);
      // ENHANCED: Include page images and target results in mock results
      setAnalysisResults({
        job_id: 'mock_job',
        signatures: mockSignatures,
        clusters: mockClusters,
        similarity_matrix: mockSimilarityMatrix,
        page_images: mockPageImages,
        target_results: mockTargetResults,
        signer_profiles: mockSignerProfiles,
        level1_signers: 2,
        level2_final_signers: 2
      });
      
      console.log('🎯 Mock analysis completed with page images and target results');
      return;
    }

    if (!jobId) {
      setError('No job ID available. Please upload files first.');
      return;
    }

    try {
      setAnalysisInProgress(true);
      setError(null);
      
      const success = await startAnalysis(jobId, analysisConfig);
      
      if (!success) {
        setAnalysisInProgress(false);
        setError('Failed to start analysis');
      } else {
        console.log('✅ Analysis started successfully');
      }
    } catch (err) {
      setAnalysisInProgress(false);
      const errorMessage = err instanceof Error ? err.message : 'Analysis failed to start';
      setError(errorMessage);
    }
  }, [demoMode, jobId, analysisConfig, startAnalysis, setError]);

  // Reset demo with cleanup
  const handleResetDemo = useCallback(async () => {
    // Cleanup backend job if exists
    if (jobId && isConnected) {
      try {
        await cleanupJob(jobId);
      } catch (err) {
        console.warn('⚠️ Cleanup warning:', err);
      }
    }

    // Reset all state
    setReferenceFile(null);
    setTargetFiles([]);
    setJobId(null);
    setAnalysisStatus(null);
    setAnalysisResults(null);
    setAnalysisInProgress(false);
    setError(null);
    
    // Reset config filename
    setAnalysisConfig(prev => ({
      ...prev,
      reference_filename: ''
    }));
    
    console.log('🔄 Demo reset complete');
  }, [jobId, isConnected, cleanupJob, setError]);

  // ENHANCED: Download report with page image data
  const handleDownloadReport = useCallback(async () => {
    if (demoMode === 'simulation') {
      // Create a mock download with page image data
      const mockData = {
        analysis_summary: {
          level1_signers: 2,
          level2_final_signers: 2,
          target_pdfs_processed: 2,
          page_images_processed: mockPageImages.length
        },
        signatures: mockSignatures,
        clusters: mockClusters,
        similarity_matrix: mockSimilarityMatrix,
        page_images: mockPageImages,
        target_results: mockTargetResults
      };
      
      const blob = new Blob([JSON.stringify(mockData, null, 2)], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'signature_analysis_report_with_crossref_mock.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      return;
    }

    if (jobId) {
      try {
        await downloadReport(jobId);
        console.log('✅ Report downloaded');
      } catch (err) {
        console.error('❌ Download failed:', err);
      }
    }
  }, [demoMode, jobId, downloadReport]);

  // Check API connection on mount
  useEffect(() => {
    checkConnection();
    
    // Auto-check connection every 30 seconds
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, [checkConnection]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.ctrlKey || e.metaKey) return;
      
      switch (e.key.toLowerCase()) {
        case 's':
          if (canStartAnalysis()) {
            handleStartAnalysis();
          }
          break;
        case 'r':
          handleResetDemo();
          break;
        case 'd':
          if (analysisResults && jobId) {
            handleDownloadReport();
          }
          break;
        case '1':
          setDemoMode('detection');
          break;
        case '2':
          setDemoMode('clustering');
          break;
        case 'f':
          setDemoMode('full');
          break;
        case 'm':
          setDemoMode('simulation');
          break;
        case 'h':
          setShowKeyboardShortcuts(!showKeyboardShortcuts);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [showKeyboardShortcuts, analysisResults, jobId, canStartAnalysis, handleStartAnalysis, handleResetDemo, handleDownloadReport]);

  // Progress polling with proper error handling
  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    
    if (analysisInProgress && jobId && demoMode !== 'simulation') {
      intervalId = setInterval(async () => {
        try {
          const status = await getAnalysisStatus(jobId);
          if (status) {
            setAnalysisStatus(status);
            
            if (status.status === 'completed') {
              setAnalysisInProgress(false);
              console.log('🎉 Analysis completed, fetching results...');
              
              const results = await getAnalysisResults(jobId);
              if (results) {
                setAnalysisResults(results);
                console.log('✅ Results fetched successfully');
                console.log(`📄 Page images received: ${results.page_images?.length || 0}`);
              }
            } else if (status.status === 'failed' || status.status === 'error') {
              setAnalysisInProgress(false);
              setError(status.error || 'Analysis failed on backend');
            }
          }
        } catch (err) {
          console.error('❌ Error polling status:', err);
          // Don't set error here to avoid interrupting analysis
        }
      }, 3000);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [analysisInProgress, jobId, getAnalysisStatus, getAnalysisResults, demoMode, setError]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-stone-50 via-stone-100 to-stone-200">
      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -inset-10 opacity-30">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-amber-100/40 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute top-3/4 right-1/4 w-96 h-96 bg-stone-200/40 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
          <div className="absolute bottom-1/4 left-1/2 w-96 h-96 bg-amber-50/60 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '4s' }}></div>
        </div>
      </div>

      <div className="relative z-10">
        <Header 
          isConnected={isConnected} 
          onConnectionCheck={checkConnection}
          loading={loading}
        />

        <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
          {/* Demo Mode Selector */}
          <div className="flex items-center justify-center space-x-4 mb-8">
            <div className="flex items-center space-x-2 bg-white/80 backdrop-blur-md rounded-lg border border-stone-300/50 p-2 shadow-sm">
              {[
                { mode: 'full' as DemoMode, label: 'Full Demo', icon: Monitor },
                { mode: 'detection' as DemoMode, label: 'Detection Only', icon: Play },
                { mode: 'clustering' as DemoMode, label: 'Clustering Only', icon: Play },
                { mode: 'simulation' as DemoMode, label: 'Simulation', icon: Play }
              ].map(({ mode, label, icon: Icon }) => (
                <button
                  key={mode}
                  onClick={() => setDemoMode(mode)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-300 ${
                    demoMode === mode
                      ? 'bg-blue-500/10 text-blue-700 border border-blue-500/20'
                      : 'text-stone-600 hover:text-stone-800 hover:bg-stone-100/50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="text-sm font-medium">{label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="text-red-700 font-medium">❌ {error}</div>
              <button 
                onClick={() => setError(null)}
                className="mt-2 text-red-600 hover:text-red-800 text-sm underline"
              >
                Dismiss
              </button>
            </div>
          )}

          {/* Connection Status */}
          {!isConnected && (
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
              <div className="text-amber-700 font-medium">
                ⚠️ Backend not connected. 
                <button 
                  onClick={checkConnection}
                  className="ml-2 text-amber-600 hover:text-amber-800 underline"
                  disabled={loading}
                >
                  {loading ? 'Checking...' : 'Retry Connection'}
                </button>
              </div>
              <div className="text-amber-600 text-sm mt-1">
                Make sure Flask server is running: <code className="bg-amber-100 px-1 rounded">python flask_backend.py</code>
              </div>
            </div>
          )}

          {/* File Upload Section */}
          {(demoMode === 'full' || demoMode === 'detection') && demoMode !== 'simulation' && (
            <FileUpload
              onReferenceUpload={handleReferenceUpload}
              onTargetUpload={handleTargetUpload}
              referenceFile={referenceFile}
              targetFiles={targetFiles}
              loading={loading}
              disabled={!isConnected}
            />
          )}

          {/* Analysis Configuration */}
          {(demoMode === 'full' || demoMode === 'simulation') && (
            <AnalysisConfiguration
              config={analysisConfig}
              onChange={setAnalysisConfig}
              disabled={analysisInProgress}
            />
          )}

          {/* Control Panel */}
          <div className="flex items-center justify-center space-x-4">
            <button
              onClick={handleStartAnalysis}
              disabled={!canStartAnalysis() || analysisInProgress}
              className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 disabled:from-stone-300 disabled:to-stone-400 text-white rounded-lg font-medium transition-all duration-300 disabled:cursor-not-allowed disabled:opacity-50 shadow-md"
            >
              <Play className="w-5 h-5" />
              <span>
                {analysisInProgress ? 'Analysis Running...' : 'Start Two-Level Analysis'}
              </span>
            </button>

            <button
              onClick={handleResetDemo}
              disabled={loading}
              className="flex items-center space-x-2 px-4 py-3 bg-stone-300 hover:bg-stone-400 disabled:bg-stone-200 text-stone-700 rounded-lg font-medium transition-all duration-300 disabled:cursor-not-allowed disabled:opacity-50 shadow-md"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Reset Demo</span>
            </button>

            <button
              onClick={() => setShowKeyboardShortcuts(!showKeyboardShortcuts)}
              className="flex items-center space-x-2 px-4 py-3 bg-purple-50 hover:bg-purple-100 border border-purple-200 text-purple-700 rounded-lg font-medium transition-all duration-300 shadow-md"
            >
              <Keyboard className="w-4 h-4" />
              <span>Shortcuts</span>
            </button>

            {/* Download Button */}
            {analysisResults && (
              <button
                onClick={handleDownloadReport}
                disabled={loading}
                className="flex items-center space-x-2 px-4 py-3 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 disabled:from-stone-300 disabled:to-stone-400 text-white rounded-lg font-medium transition-all duration-300 disabled:cursor-not-allowed disabled:opacity-50 shadow-md"
              >
                <Download className="w-4 h-4" />
                <span>Download Report</span>
              </button>
            )}
          </div>

          {/* Keyboard Shortcuts Modal */}
          {showKeyboardShortcuts && (
            <div className="fixed inset-0 bg-black/20 backdrop-blur-sm z-50 flex items-center justify-center p-4">
              <div className="bg-white border border-stone-200 rounded-xl p-6 max-w-md w-full shadow-xl">
                <h3 className="text-xl font-semibold text-stone-800 mb-4">⌨️ Keyboard Shortcuts</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between items-center">
                    <span className="text-stone-600">Start Analysis</span>
                    <kbd className="bg-stone-200 px-2 py-1 rounded text-stone-800 font-mono">S</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-stone-600">Reset Demo</span>
                    <kbd className="bg-stone-200 px-2 py-1 rounded text-stone-800 font-mono">R</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-stone-600">Download Report</span>
                    <kbd className="bg-stone-200 px-2 py-1 rounded text-stone-800 font-mono">D</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-stone-600">Full Demo Mode</span>
                    <kbd className="bg-stone-200 px-2 py-1 rounded text-stone-800 font-mono">F</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-stone-600">Detection Mode</span>
                    <kbd className="bg-stone-200 px-2 py-1 rounded text-stone-800 font-mono">1</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-stone-600">Clustering Mode</span>
                    <kbd className="bg-stone-200 px-2 py-1 rounded text-stone-800 font-mono">2</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-stone-600">Simulation Mode</span>
                    <kbd className="bg-stone-200 px-2 py-1 rounded text-stone-800 font-mono">M</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-stone-600">Toggle Shortcuts</span>
                    <kbd className="bg-stone-200 px-2 py-1 rounded text-stone-800 font-mono">H</kbd>
                  </div>
                </div>
                <button
                  onClick={() => setShowKeyboardShortcuts(false)}
                  className="mt-6 w-full px-4 py-2 bg-blue-50 hover:bg-blue-100 text-blue-700 border border-blue-200 rounded-lg transition-colors duration-300"
                >
                  Close
                </button>
              </div>
            </div>
          )}

          {/* Progress Tracking */}
          {analysisStatus && (demoMode === 'full' || demoMode === 'simulation') && (
            <ProgressTracker 
              status={analysisStatus} 
              demoMode={demoMode}
            />
          )}

          {/* Results Sections */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
            {/* ENHANCED: Signature Detection with Page Images */}
            {(demoMode === 'full' || demoMode === 'detection' || demoMode === 'simulation') && (
              <SignatureDetection
                signatures={analysisResults?.signatures || (demoMode === 'simulation' ? mockSignatures : [])}
                pageImages={analysisResults?.page_images || (demoMode === 'simulation' ? mockPageImages : [])}
                loading={analysisInProgress && demoMode !== 'simulation'}
              />
            )}

            {/* Clustering Animation */}
            {(demoMode === 'full' || demoMode === 'clustering' || demoMode === 'simulation') && (
              <ClusteringAnimation
                clusters={analysisResults?.clusters || (demoMode === 'simulation' ? mockClusters : [])}
                signatures={analysisResults?.signatures || (demoMode === 'simulation' ? mockSignatures : [])}
                loading={analysisInProgress && demoMode !== 'simulation'}
                demoMode={demoMode}
              />
            )}
          </div>

          {/* Similarity Matrix */}
          {(demoMode === 'full' || demoMode === 'simulation') && (
            <SimilarityMatrix
              similarityMatrix={analysisResults?.similarity_matrix || (demoMode === 'simulation' ? mockSimilarityMatrix : [])}
              signatures={analysisResults?.signatures || (demoMode === 'simulation' ? mockSignatures : [])}
              loading={analysisInProgress && demoMode !== 'simulation'}
              demoMode={demoMode}
            />
          )}

          {/* ENHANCED: Results Dashboard with Page Image Support */}
          {(demoMode === 'full' || demoMode === 'simulation') && (
            <ResultsDashboard
              results={analysisResults || (demoMode === 'simulation' ? {
                job_id: 'mock_job',
                signatures: mockSignatures,
                clusters: mockClusters,
                similarity_matrix: mockSimilarityMatrix,
                page_images: mockPageImages,
                level1_signers: 2,
                level2_final_signers: 2,
                target_results: mockTargetResults,
                signer_profiles: mockSignerProfiles,
                report_path: 'mock_report.xlsx'
              } : null)}
              onDownloadReport={handleDownloadReport}
              loading={loading}
              demoMode={demoMode}
              jobId={jobId}
            />
          )}

          {/* NEW: Cross-Reference Results Section */}
          {analysisResults && (
            <div className="space-y-8">
              {/* Reference PDF Analysis Section */}
              <div>
                <h2 className="text-2xl font-bold text-stone-800 mb-6">
                  📄 Reference PDF Analysis
                </h2>
                <p className="text-stone-600 mb-6">
                  Clustering and similarity analysis of signatures from the reference document
                </p>
                {/* Reference analysis components are already displayed above */}
              </div>
              
              {/* Cross-Reference Results Section */}
              {analysisResults.target_results && Object.keys(analysisResults.target_results).length > 0 && (
                <div>
                  <h2 className="text-2xl font-bold text-stone-800 mb-6">
                    🔄 Cross-Reference Analysis
                  </h2>
                  <p className="text-stone-600 mb-6">
                    Comparison results showing which reference signers were found in target PDFs
                  </p>
                  <CrossReferenceResults 
                    targetResults={analysisResults.target_results}
                    signerProfiles={analysisResults.signer_profiles || {}}
                  />
                </div>
              )}
              
              {/* Show message if no target results */}
              {(!analysisResults.target_results || Object.keys(analysisResults.target_results).length === 0) && (
                <div className="bg-amber-50 border border-amber-200 rounded-lg p-6 text-center">
                  <h3 className="text-lg font-medium text-amber-800 mb-2">📋 No Cross-Reference Results</h3>
                  <p className="text-amber-700">
                    Upload target PDFs and run analysis to see cross-reference comparison results.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* ENHANCED: Demo Instructions with Page View Information */}
          {!analysisInProgress && !analysisResults && (
            <div className="bg-white/80 backdrop-blur-md border border-stone-200 rounded-xl p-6 shadow-sm">
              <h3 className="text-lg font-semibold text-blue-700 mb-4">🎮 Demo Instructions</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-stone-700">
                <div>
                  <h4 className="font-medium text-stone-800 mb-2">For Live Demo:</h4>
                  <ul className="space-y-1">
                    <li>• Ensure Flask backend is running</li>
                    <li>• Upload reference PDF with signatures</li>
                    <li>• Select multiple target PDFs</li>
                    <li>• Start analysis and watch real-time progress</li>
                    <li>• NEW: View cross-reference comparison results</li>
                    <li>• NEW: See signatures in full page context</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-medium text-stone-800 mb-2">For Presentation:</h4>
                  <ul className="space-y-1">
                    <li>• Use Simulation mode for reliable demo</li>
                    <li>• Press 'H' to see all keyboard shortcuts</li>
                    <li>• Different modes focus on specific features</li>
                    <li>• Click elements for detailed explanations</li>
                    <li>• NEW: Demo includes cross-reference results</li>
                    <li>• NEW: Interactive signature overlays with animations</li>
                  </ul>
                </div>
              </div>
              
              {/* NEW: Cross-Reference Feature Highlight */}
              <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <h5 className="font-medium text-green-700">Key Feature: Cross-Reference Analysis</h5>
                </div>
                <p className="text-green-700 text-sm">
                  The system now displays complete cross-reference results! See which reference signers appear 
                  in each target PDF, with match counts, similarity percentages, and detailed signature-level analysis. 
                  Perfect for document verification and fraud detection workflows.
                </p>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;