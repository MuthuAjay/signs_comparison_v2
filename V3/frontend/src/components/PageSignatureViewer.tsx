import React, { useState, useRef, useEffect, useCallback } from 'react';
import { PageImage, BoundingBoxOverlay, SignatureSelection, AnimationConfig } from '../types';

interface PageSignatureViewerProps {
  pageImage: PageImage;
  selectedSignatureIds: string[];
  hoveredSignatureId?: string;
  onSignatureSelect: (signatureId: string) => void;
  onSignatureHover: (signatureId: string | undefined) => void;
  animationConfig?: AnimationConfig;
  className?: string;
  enableZoom?: boolean;
  maxZoomLevel?: number;
}

interface ZoomState {
  scale: number;
  translateX: number;
  translateY: number;
}

const PageSignatureViewer: React.FC<PageSignatureViewerProps> = ({
  pageImage,
  selectedSignatureIds,
  hoveredSignatureId,
  onSignatureSelect,
  onSignatureHover,
  animationConfig = {
    enablePulse: true,
    enableHover: true,
    enableClick: true,
    confidenceColorMapping: true,
    animationSpeed: 'medium'
  },
  className = '',
  enableZoom = true,
  maxZoomLevel = 3
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [imageDisplaySize, setImageDisplaySize] = useState({ width: 0, height: 0 });
  const [zoomState, setZoomState] = useState<ZoomState>({ scale: 1, translateX: 0, translateY: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0, translateX: 0, translateY: 0 });

  // Calculate confidence-based color
  const getConfidenceColor = useCallback((confidence: number): string => {
    if (!animationConfig.confidenceColorMapping) return '#3b82f6';
    
    if (confidence >= 0.9) return '#22c55e'; // Green - High confidence
    if (confidence >= 0.7) return '#eab308'; // Yellow - Medium confidence
    if (confidence >= 0.5) return '#f97316'; // Orange - Low confidence
    return '#ef4444'; // Red - Very low confidence
  }, [animationConfig.confidenceColorMapping]);

  // Get animation duration based on speed setting
  const getAnimationDuration = useCallback((): string => {
    switch (animationConfig.animationSpeed) {
      case 'slow': return '2s';
      case 'fast': return '0.8s';
      default: return '1.4s';
    }
  }, [animationConfig.animationSpeed]);

  // Update container size on resize
// Update container size detection:
 // Update container size detection:
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        console.log('Container rect:', rect); // Debug log
        setContainerSize({ 
          width: rect.width || 800,   // Fallback values
          height: rect.height || 600 
        });
      }
    };

    // Force initial measurement
    setTimeout(updateSize, 100); // Delay to ensure DOM is ready
    updateSize();
    
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []); // Remove dependencies that might cause loops

// Add this right after the imageLoaded state
console.log('Image Loading State:', {
  imageLoaded,
  hasImageData: !!pageImage.image_data,
  imageDataLength: pageImage.image_data?.length,
  containerSize,
  imageDisplaySize
});
  // Calculate image display size maintaining aspect ratio
  useEffect(() => {
    if (imageLoaded && containerSize.width > 0 && containerSize.height > 0) {
      const imageAspectRatio = pageImage.dimensions.width / pageImage.dimensions.height;
      const containerAspectRatio = containerSize.width / containerSize.height;

      let displayWidth, displayHeight;

      if (imageAspectRatio > containerAspectRatio) {
        // Image is wider than container
        displayWidth = containerSize.width;
        displayHeight = containerSize.width / imageAspectRatio;
      } else {
        // Image is taller than container
        displayHeight = containerSize.height;
        displayWidth = containerSize.height * imageAspectRatio;
      }

      setImageDisplaySize({ width: displayWidth, height: displayHeight });
    }
  }, [imageLoaded, containerSize, pageImage.dimensions]);

  // Convert bounding box coordinates to display coordinates
  const convertBoundingBox = useCallback((bbox: number[]): BoundingBoxOverlay => {
    if (!imageLoaded || imageDisplaySize.width === 0) {
      return { id: '', x: 0, y: 0, width: 0, height: 0, confidence_score: 0 };
    }

    const [x1, y1, x2, y2] = bbox;
    
    // Scale from original image dimensions to display dimensions
    const scaleX = imageDisplaySize.width / pageImage.dimensions.width;
    const scaleY = imageDisplaySize.height / pageImage.dimensions.height;

    return {
      id: '',
      x: x1 * scaleX,
      y: y1 * scaleY,
      width: (x2 - x1) * scaleX,
      height: (y2 - y1) * scaleY,
      confidence_score: 0
    };
  }, [imageLoaded, imageDisplaySize, pageImage.dimensions]);

  // Handle signature click with zoom functionality
  const handleSignatureClick = useCallback((signature: any) => {
    onSignatureSelect(signature.unique_id);

    if (enableZoom && animationConfig.enableClick) {
      const overlay = convertBoundingBox(signature.bounding_box);
      const centerX = overlay.x + overlay.width / 2;
      const centerY = overlay.y + overlay.height / 2;

      // Calculate zoom to fit signature with some padding
      const zoomScale = Math.min(
        imageDisplaySize.width / (overlay.width * 2),
        imageDisplaySize.height / (overlay.height * 2),
        maxZoomLevel
      );

      // Calculate translation to center the signature
      const translateX = (imageDisplaySize.width / 2 - centerX * zoomScale);
      const translateY = (imageDisplaySize.height / 2 - centerY * zoomScale);

      setZoomState({ scale: zoomScale, translateX, translateY });
    }
  }, [onSignatureSelect, enableZoom, animationConfig.enableClick, convertBoundingBox, imageDisplaySize, maxZoomLevel]);

  // Handle mouse events for dragging
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (zoomState.scale > 1) {
      setIsDragging(true);
      setDragStart({
        x: e.clientX,
        y: e.clientY,
        translateX: zoomState.translateX,
        translateY: zoomState.translateY
      });
    }
  }, [zoomState]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDragging) {
      const deltaX = e.clientX - dragStart.x;
      const deltaY = e.clientY - dragStart.y;
      
      setZoomState(prev => ({
        ...prev,
        translateX: dragStart.translateX + deltaX,
        translateY: dragStart.translateY + deltaY
      }));
    }
  }, [isDragging, dragStart]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Reset zoom on double click
  const handleDoubleClick = useCallback(() => {
    setZoomState({ scale: 1, translateX: 0, translateY: 0 });
  }, []);

  // Handle wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (enableZoom) {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      const newScale = Math.max(1, Math.min(maxZoomLevel, zoomState.scale * delta));
      
      if (newScale !== zoomState.scale) {
        setZoomState(prev => ({ ...prev, scale: newScale }));
      }
    }
  }, [enableZoom, maxZoomLevel, zoomState.scale]);

  // In PageSignatureViewer, add:   
  console.log('Page Image Data Length:', pageImage.image_data?.length);
  console.log('Image Data Preview:', pageImage.image_data?.substring(0, 50));

  return (
    <div className={`page-signature-viewer ${className}`}>
      {/* Page Info Header */}
      <div className="flex justify-between items-center p-4 bg-stone-50 border-b border-stone-200">
        <div>
          <h3 className="font-semibold text-stone-800">{pageImage.pdf_name}</h3>
          <p className="text-sm text-stone-600">
            Page {pageImage.page_number} • {pageImage.signatures.length} signature(s) detected
          </p>
        </div>
        <div className="flex items-center gap-2">
          {zoomState.scale > 1 && (
            <button
              onClick={() => setZoomState({ scale: 1, translateX: 0, translateY: 0 })}
              className="px-3 py-1 text-sm bg-amber-600 text-white rounded hover:bg-amber-700 transition-colors shadow-sm"
            >
              Reset Zoom
            </button>
          )}
          <span className="text-sm text-stone-600">
            {Math.round(zoomState.scale * 100)}%
          </span>
        </div>
      </div>

      {/* Page Image Container */}
      <div 
        ref={containerRef}
        className="relative flex-1 overflow-hidden bg-stone-100 cursor-move"
        style={{ height: '600px', minHeight: '400px' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onDoubleClick={handleDoubleClick}
        onWheel={handleWheel}
        // style={{ height: '600px' }}
      >
        {/* Page Image */}
        <div 
          className="relative mx-auto"
          style={{
            width: imageDisplaySize.width,
            height: imageDisplaySize.height,
            transform: `scale(${zoomState.scale}) translate(${zoomState.translateX / zoomState.scale}px, ${zoomState.translateY / zoomState.scale}px)`,
            transformOrigin: 'center center',
            transition: isDragging ? 'none' : 'transform 0.3s ease-out'
          }}
        >
          {pageImage.image_data && (
            <img
              ref={imageRef}
              src={pageImage.image_data}
              alt={`Page ${pageImage.page_number} of ${pageImage.pdf_name}`}
              className="w-full h-full object-contain shadow-lg"
              onLoad={() => setImageLoaded(true)}
              onError={() => setImageLoaded(false)}
              draggable={false}
            />
          )}

          {/* Signature Bounding Box Overlays */}
          {imageLoaded && pageImage.signatures.map((signature) => {
            const overlay = convertBoundingBox(signature.bounding_box);
            const isSelected = selectedSignatureIds.includes(signature.unique_id);
            const isHovered = hoveredSignatureId === signature.unique_id;
            const confidenceColor = getConfidenceColor(signature.confidence_score);

            return (
              <div
                key={signature.unique_id}
                className={`absolute border-2 cursor-pointer transition-all duration-200 ${
                  isSelected ? 'border-amber-600 shadow-lg' : 'border-transparent'
                } ${isHovered ? 'shadow-md' : ''}`}
                style={{
                  left: overlay.x,
                  top: overlay.y,
                  width: overlay.width,
                  height: overlay.height,
                  borderColor: isSelected ? '#d97706' : confidenceColor,
                  borderWidth: isSelected ? '3px' : '2px',
                  backgroundColor: isSelected ? 'rgba(217, 119, 6, 0.1)' : 'rgba(255, 255, 255, 0.05)',
                  boxShadow: isHovered ? `0 0 15px ${confidenceColor}` : 'none'
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  handleSignatureClick(signature);
                }}
                onMouseEnter={() => animationConfig.enableHover && onSignatureHover(signature.unique_id)}
                onMouseLeave={() => animationConfig.enableHover && onSignatureHover(undefined)}
              >
                {/* Animated Corners */}
                {animationConfig.enablePulse && (
                  <>
                    <div 
                      className="absolute w-2 h-2 border-l-2 border-t-2 -top-1 -left-1"
                      style={{ 
                        borderColor: confidenceColor,
                        animation: `pulse ${getAnimationDuration()} infinite`
                      }}
                    />
                    <div 
                      className="absolute w-2 h-2 border-r-2 border-t-2 -top-1 -right-1"
                      style={{ 
                        borderColor: confidenceColor,
                        animation: `pulse ${getAnimationDuration()} infinite 0.2s`
                      }}
                    />
                    <div 
                      className="absolute w-2 h-2 border-l-2 border-b-2 -bottom-1 -left-1"
                      style={{ 
                        borderColor: confidenceColor,
                        animation: `pulse ${getAnimationDuration()} infinite 0.4s`
                      }}
                    />
                    <div 
                      className="absolute w-2 h-2 border-r-2 border-b-2 -bottom-1 -right-1"
                      style={{ 
                        borderColor: confidenceColor,
                        animation: `pulse ${getAnimationDuration()} infinite 0.6s`
                      }}
                    />
                  </>
                )}

                {/* Confidence Score Badge */}
                <div 
                  className="absolute -top-6 left-0 px-2 py-1 text-xs font-medium text-white rounded shadow-lg"
                  style={{ backgroundColor: confidenceColor }}
                >
                  {Math.round(signature.confidence_score * 100)}%
                </div>
              </div>
            );
          })}
        </div>

        {/* Loading Overlay */}
        {!imageLoaded && (
          <div className="absolute inset-0 flex items-center justify-center bg-stone-50">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-amber-600 mx-auto mb-2"></div>
              <p className="text-stone-600">Loading page image...</p>
            </div>
          </div>
        )}
      </div>

      {/* Instructions */}
      <div className="p-3 bg-stone-50 border-t border-stone-200 text-sm text-stone-600">
        <div className="flex flex-wrap gap-4">
          <span>💡 Click signatures to select and zoom</span>
          <span>🔍 Scroll to zoom in/out</span>
          <span>🖱️ Drag to pan when zoomed</span>
          <span>⏯️ Double-click to reset zoom</span>
        </div>
      </div>

      <style jsx>{`
        @keyframes pulse {
          0%, 100% { 
            opacity: 1; 
            transform: scale(1); 
          }
          50% { 
            opacity: 0.7; 
            transform: scale(1.1); 
          }
        }
        
        .page-signature-viewer {
          display: flex;
          flex-direction: column;
          height: 100%;
          border: 1px solid #d6d3d1;
          border-radius: 8px;
          overflow: hidden;
          background: white;
        }
        
        .cursor-move {
          cursor: ${isDragging ? 'grabbing' : 'grab'};
        }
      `}</style>
    </div>
  );
};

export default PageSignatureViewer;