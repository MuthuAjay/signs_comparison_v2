import React, { useState } from 'react';
import { Search, Eye, Info, Grid3X3, FileImage, ToggleLeft, ToggleRight } from 'lucide-react';
import { SignatureDetection, PageImage, ViewMode } from '../types';
import PageSignatureViewer from './PageSignatureViewer';

interface SignatureDetectionProps {
  signatures: SignatureDetection[];
  pageImages?: PageImage[]; // NEW: Page images for page view
  loading?: boolean;
}

const SignatureDetectionComponent: React.FC<SignatureDetectionProps> = ({
  signatures,
  pageImages = [],
  loading = false
}) => {
  const [selectedSignatureIds, setSelectedSignatureIds] = useState<string[]>([]);
  const [hoveredSignatureId, setHoveredSignatureId] = useState<string | undefined>(undefined);
  const [selectedSignature, setSelectedSignature] = useState<SignatureDetection | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('grid'); // NEW: View mode state
  const [selectedPageIndex, setSelectedPageIndex] = useState<number>(0); // NEW: For page navigation

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-700 border-green-600';
    if (confidence >= 0.6) return 'text-amber-700 border-amber-600';
    return 'text-red-700 border-red-600';
  };

  const formatConfidence = (confidence: number) => {
    return (confidence * 100).toFixed(1);
  };

  // NEW: Handle signature selection (works for both views)
  const handleSignatureSelect = (signatureId: string) => {
    const isSelected = selectedSignatureIds.includes(signatureId);
    
    if (isSelected) {
      setSelectedSignatureIds(prev => prev.filter(id => id !== signatureId));
      setSelectedSignature(null);
    } else {
      setSelectedSignatureIds(prev => [...prev, signatureId]);
      const signature = signatures.find(s => s.id === signatureId);
      setSelectedSignature(signature || null);
    }
  };

  // NEW: Handle signature hover
  const handleSignatureHover = (signatureId: string | undefined) => {
    setHoveredSignatureId(signatureId);
  };

  // NEW: Switch view mode with smooth transition
  const handleViewModeChange = (newMode: ViewMode) => {
    setViewMode(newMode);
    // When switching to page view, navigate to page with first selected signature
    if (newMode === 'page' && selectedSignatureIds.length > 0 && pageImages.length > 0) {
      const selectedSig = signatures.find(s => selectedSignatureIds.includes(s.id));
      if (selectedSig) {
        const pageIndex = pageImages.findIndex(p => p.page_number === selectedSig.page_number);
        if (pageIndex >= 0) {
          setSelectedPageIndex(pageIndex);
        }
      }
    }
  };

  // NEW: Get unique page numbers from signatures for page navigation
  const getAvailablePages = () => {
    if (pageImages.length > 0) return pageImages;
    
    // Fallback: create page data from signatures if pageImages not available
    const pageNumbers = [...new Set(signatures.map(s => s.page_number))].sort((a, b) => a - b);
    return pageNumbers.map(pageNum => ({
      pdf_name: signatures.find(s => s.page_number === pageNum)?.pdf_name || 'Document',
      page_number: pageNum,
      image_data: '', // Will show placeholder if no image data
      dimensions: { width: 800, height: 1000 },
      signatures: signatures
        .filter(s => s.page_number === pageNum)
        .map(s => ({
          unique_id: s.id,
          bounding_box: [s.bounding_box.x, s.bounding_box.y, s.bounding_box.x + s.bounding_box.width, s.bounding_box.y + s.bounding_box.height],
          confidence_score: s.confidence_score,
          bbox_coordinates: s.bbox_coordinates
        }))
    }));
  };

  const availablePages = getAvailablePages();
  
  // Add this in your component to see what's happening:
  console.log('Available Pages:', availablePages);
  console.log('Page Images Prop:', pageImages);

  return (
    <div className="bg-white/90 backdrop-blur-md rounded-xl border border-stone-200 p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Search className="w-6 h-6 text-amber-700" />
          <h3 className="text-xl font-semibold text-stone-800">Signature Detection</h3>
        </div>
        
        <div className="flex items-center space-x-4">
          {/* NEW: View Mode Toggle */}
          <div className="flex items-center space-x-2 bg-stone-100 rounded-lg p-1 border border-stone-300">
            <button
              onClick={() => handleViewModeChange('grid')}
              className={`flex items-center space-x-2 px-3 py-2 rounded-md transition-all duration-200 ${
                viewMode === 'grid'
                  ? 'bg-amber-600 text-white shadow-lg'
                  : 'text-stone-600 hover:text-amber-700 hover:bg-stone-200'
              }`}
            >
              <Grid3X3 className="w-4 h-4" />
              <span className="text-sm font-medium">Grid View</span>
            </button>
            <button
              onClick={() => handleViewModeChange('page')}
              className={`flex items-center space-x-2 px-3 py-2 rounded-md transition-all duration-200 ${
                viewMode === 'page'
                  ? 'bg-amber-600 text-white shadow-lg'
                  : 'text-stone-600 hover:text-amber-700 hover:bg-stone-200'
              } ${pageImages.length === 0 ? 'opacity-50 cursor-not-allowed' : ''}`}
              disabled={pageImages.length === 0}
            >
              <FileImage className="w-4 h-4" />
              <span className="text-sm font-medium">Page View</span>
            </button>
          </div>

          <div className="flex items-center space-x-4 text-sm text-stone-600">
            <span>Found: {signatures.length} signatures</span>
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="flex items-center space-x-1 text-amber-700 hover:text-amber-800 transition-colors"
            >
              <Eye className="w-4 h-4" />
              <span>{showDetails ? 'Hide' : 'Show'} Details</span>
            </button>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-amber-600"></div>
          <span className="ml-4 text-stone-600">Detecting signatures...</span>
        </div>
      ) : signatures.length > 0 ? (
        <div className="space-y-4">
          {/* Detection Summary */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-gradient-to-r from-green-50 to-green-100 border border-green-200 rounded-lg p-4">
              <div className="text-green-800 text-2xl font-bold">
                {signatures.filter(s => s.confidence_score >= 0.8).length}
              </div>
              <div className="text-green-700 text-sm">High Confidence</div>
              <div className="text-green-600 text-xs">≥ 80%</div>
            </div>
            <div className="bg-gradient-to-r from-amber-50 to-amber-100 border border-amber-200 rounded-lg p-4">
              <div className="text-amber-800 text-2xl font-bold">
                {signatures.filter(s => s.confidence_score >= 0.6 && s.confidence_score < 0.8).length}
              </div>
              <div className="text-amber-700 text-sm">Medium Confidence</div>
              <div className="text-amber-600 text-xs">60-79%</div>
            </div>
            <div className="bg-gradient-to-r from-red-50 to-red-100 border border-red-200 rounded-lg p-4">
              <div className="text-red-800 text-2xl font-bold">
                {signatures.filter(s => s.confidence_score < 0.6).length}
              </div>
              <div className="text-red-700 text-sm">Low Confidence</div>
              <div className="text-red-600 text-xs">&lt; 60%</div>
            </div>
          </div>

          {/* NEW: View Content with Smooth Transitions */}
          <div className="transition-all duration-300 ease-in-out">
            {viewMode === 'grid' ? (
              /* EXISTING: Grid View */
              <div className="animate-fadeIn">
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  {signatures.map((signature) => (
                    <div
                      key={signature.id}
                      className={`relative group cursor-pointer transition-all duration-300 ${
                        selectedSignatureIds.includes(signature.id)
                          ? 'ring-2 ring-amber-600 scale-105'
                          : 'hover:scale-105 hover:ring-1 hover:ring-amber-600/50'
                      } ${hoveredSignatureId === signature.id ? 'ring-1 ring-amber-500' : ''}`}
                      onClick={() => handleSignatureSelect(signature.id)}
                      onMouseEnter={() => handleSignatureHover(signature.id)}
                      onMouseLeave={() => handleSignatureHover(undefined)}
                    >
                      <div className="bg-white/80 rounded-lg p-3 border border-stone-300 shadow-sm">
                        {/* Signature Preview */}
                        <div className="aspect-square bg-stone-100 rounded-md mb-2 flex items-center justify-center">
                          {signature.image_data ? (
                            <img
                              src={signature.image_data}
                              alt={`Signature ${signature.id}`}
                              className="max-w-full max-h-full object-contain rounded"
                            />
                          ) : (
                            <div className="text-stone-600 text-xs text-center">
                              <Search className="w-6 h-6 mx-auto mb-1" />
                              Signature Preview
                            </div>
                          )}
                        </div>

                        {/* Signature Info */}
                        <div className="space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="text-stone-800 text-xs font-medium">
                              Page {signature.page_number}
                            </span>
                            <span
                              className={`text-xs px-2 py-1 rounded border ${getConfidenceColor(
                                signature.confidence_score
                              )}`}
                            >
                              {formatConfidence(signature.confidence_score)}%
                            </span>
                          </div>
                          
                          {showDetails && (
                            <div className="text-xs text-stone-600 space-y-1">
                              <div>Size: {signature.bounding_box.width}×{signature.bounding_box.height}</div>
                              <div>Pos: ({signature.bounding_box.x}, {signature.bounding_box.y})</div>
                            </div>
                          )}
                        </div>

                        {/* Animated Detection Box */}
                        <div className="absolute inset-0 pointer-events-none">
                          <div className="absolute inset-2 border-2 border-amber-600 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                            <div className="absolute -top-1 -left-1 w-2 h-2 bg-amber-600 rounded-full animate-pulse"></div>
                            <div className="absolute -top-1 -right-1 w-2 h-2 bg-amber-600 rounded-full animate-pulse"></div>
                            <div className="absolute -bottom-1 -left-1 w-2 h-2 bg-amber-600 rounded-full animate-pulse"></div>
                            <div className="absolute -bottom-1 -right-1 w-2 h-2 bg-amber-600 rounded-full animate-pulse"></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              /* NEW: Page View */
              <div className="animate-fadeIn">
                {availablePages.length > 0 ? (
                  <div className="space-y-4">
                    {/* Page Navigation */}
                    {availablePages.length > 1 && (
                      <div className="flex items-center justify-center space-x-2 mb-4">
                        <span className="text-stone-600 text-sm">Page:</span>
                        {availablePages.map((page, index) => (
                          <button
                            key={`page-${page.pdf_name}-${page.page_number}-${index}`}  // Unique keys
                            onClick={() => setSelectedPageIndex(index)}
                            className={`px-3 py-1 rounded-md text-sm transition-all duration-200 ${
                              selectedPageIndex === index
                                ? 'bg-amber-600 text-white'
                                : 'bg-stone-200 text-stone-600 hover:bg-stone-300 hover:text-amber-700'
                            }`}
                          >
                            {page.page_number}
                          </button>
                        ))}
                      </div>
                    )}

                    {/* Page Signature Viewer */}
                    <div className="bg-stone-50 rounded-lg border border-stone-300 overflow-hidden shadow-inner">
                      <PageSignatureViewer
                        pageImage={availablePages[selectedPageIndex]}
                        selectedSignatureIds={selectedSignatureIds}
                        hoveredSignatureId={hoveredSignatureId}
                        onSignatureSelect={handleSignatureSelect}
                        onSignatureHover={handleSignatureHover}
                        animationConfig={{
                          enablePulse: true,
                          enableHover: true,
                          enableClick: true,
                          confidenceColorMapping: true,
                          animationSpeed: 'medium'
                        }}
                        enableZoom={true}
                        maxZoomLevel={5}
                        className="h-full min-h-[600px]" 
                        
                      />
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12 text-stone-600 bg-stone-50 rounded-lg border border-stone-300 shadow-inner">
                    <FileImage className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p>No page images available</p>
                    <p className="text-sm mt-2">Page view requires full page image data from the backend</p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Selected Signature Details */}
          {selectedSignature && (
            <div className="mt-6 p-6 bg-stone-50/80 rounded-lg border border-amber-200 shadow-inner animate-fadeIn">
              <div className="flex items-center space-x-3 mb-4">
                <Info className="w-5 h-5 text-amber-700" />
                <h4 className="text-lg font-semibold text-stone-800">
                  Signature Details - {selectedSignature.id}
                </h4>
                <span className="text-sm text-stone-600">
                  (Selected in {viewMode === 'grid' ? 'Grid' : 'Page'} View)
                </span>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-stone-600">Page Number:</span>
                    <span className="text-stone-800 font-medium">{selectedSignature.page_number}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-stone-600">Confidence Score:</span>
                    <span className={`font-medium ${getConfidenceColor(selectedSignature.confidence_score).split(' ')[0]}`}>
                      {formatConfidence(selectedSignature.confidence_score)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-stone-600">Width:</span>
                    <span className="text-stone-800 font-medium">{selectedSignature.bounding_box.width}px</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-stone-600">Height:</span>
                    <span className="text-stone-800 font-medium">{selectedSignature.bounding_box.height}px</span>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-stone-600">X Position:</span>
                    <span className="text-stone-800 font-medium">{selectedSignature.bounding_box.x}px</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-stone-600">Y Position:</span>
                    <span className="text-stone-800 font-medium">{selectedSignature.bounding_box.y}px</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-stone-600">Aspect Ratio:</span>
                    <span className="text-stone-800 font-medium">
                      {(selectedSignature.bounding_box.width / selectedSignature.bounding_box.height).toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-stone-600">Area:</span>
                    <span className="text-stone-800 font-medium">
                      {(selectedSignature.bounding_box.width * selectedSignature.bounding_box.height).toLocaleString()}px²
                    </span>
                  </div>
                </div>
              </div>

              {/* NEW: Quick View Switch Button */}
              <div className="mt-4 pt-4 border-t border-stone-300">
                <button
                  onClick={() => handleViewModeChange(viewMode === 'grid' ? 'page' : 'grid')}
                  className="flex items-center space-x-2 text-amber-700 hover:text-amber-800 transition-colors text-sm"
                  disabled={viewMode === 'page' && pageImages.length === 0}
                >
                  {viewMode === 'grid' ? <FileImage className="w-4 h-4" /> : <Grid3X3 className="w-4 h-4" />}
                  <span>
                    Switch to {viewMode === 'grid' ? 'Page' : 'Grid'} View
                  </span>
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="text-center py-12 text-stone-600">
          <Search className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p>No signatures detected yet</p>
          <p className="text-sm mt-2">Upload files and start analysis to see results</p>
        </div>
      )}

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }
      `}</style>
    </div>
  );
};

export default SignatureDetectionComponent;