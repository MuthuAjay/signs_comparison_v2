import React, { useState } from 'react';
import { Grid, Info, Eye } from 'lucide-react';
import { SimilarityScore, SignatureDetection } from '../types';

interface SimilarityMatrixProps {
  similarityMatrix: SimilarityScore[];
  signatures: SignatureDetection[];
  loading?: boolean;
}

const SimilarityMatrix: React.FC<SimilarityMatrixProps> = ({
  similarityMatrix,
  signatures,
  loading = false
}) => {
  const [selectedCell, setSelectedCell] = useState<SimilarityScore | null>(null);
  const [showTooltip, setShowTooltip] = useState<{ x: number; y: number; score: SimilarityScore } | null>(null);

  const getSimilarityColor = (similarity: number): string => {
    if (similarity >= 0.8) return 'bg-green-500';
    if (similarity >= 0.6) return 'bg-yellow-500';
    if (similarity >= 0.4) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const getSimilarityIntensity = (similarity: number): string => {
    const opacity = Math.max(0.2, similarity);
    return `opacity-${Math.round(opacity * 100)}`;
  };

  const formatFeatureScore = (score: number): string => {
    return (score * 100).toFixed(1);
  };

  // Create matrix grid
  const matrixSize = signatures.length;
  const matrix = Array(matrixSize).fill(null).map(() => Array(matrixSize).fill(null));

  // Fill matrix with similarity scores
  similarityMatrix.forEach(score => {
    const index1 = signatures.findIndex(s => s.id === score.signature1_id);
    const index2 = signatures.findIndex(s => s.id === score.signature2_id);
    
    if (index1 !== -1 && index2 !== -1) {
      matrix[index1][index2] = score;
      matrix[index2][index1] = score; // Make symmetric
    }
  });

  // Fill diagonal with perfect matches
  for (let i = 0; i < matrixSize; i++) {
    matrix[i][i] = {
      signature1_id: signatures[i].id,
      signature2_id: signatures[i].id,
      similarity: 1.0,
      features: { hog: 1.0, resnet50: 1.0, vgg19: 1.0, vit: 1.0 }
    } as SimilarityScore;
  }

  return (
    <div className="bg-white/90 backdrop-blur-md rounded-xl border border-stone-200 p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Grid className="w-6 h-6 text-green-700" />
          <h3 className="text-xl font-semibold text-stone-800">Similarity Matrix</h3>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 text-sm">
            <div className="flex space-x-1">
              <div className="w-3 h-3 bg-green-500 rounded"></div>
              <span className="text-stone-600">High (≥80%)</span>
            </div>
            <div className="flex space-x-1">
              <div className="w-3 h-3 bg-yellow-500 rounded"></div>
              <span className="text-stone-600">Med (60-79%)</span>
            </div>
            <div className="flex space-x-1">
              <div className="w-3 h-3 bg-orange-500 rounded"></div>
              <span className="text-stone-600">Low (40-59%)</span>
            </div>
            <div className="flex space-x-1">
              <div className="w-3 h-3 bg-red-500 rounded"></div>
              <span className="text-stone-600">Poor (&lt;40%)</span>
            </div>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-600"></div>
          <span className="ml-4 text-stone-600">Computing similarity matrix...</span>
        </div>
      ) : signatures.length > 0 ? (
        <div className="space-y-6">
          {/* Matrix */}
          <div className="overflow-auto">
            <div className="inline-block min-w-full">
              {/* Header row with signature previews */}
              <div className="flex mb-2">
                <div className="w-16 h-16 flex items-center justify-center">
                  <span className="text-stone-600 text-xs font-medium">vs</span>
                </div>
                {signatures.map((sig, index) => (
                  <div key={sig.id} className="w-16 h-16 p-1">
                    <div className="w-full h-full bg-stone-100 rounded border border-stone-300 flex items-center justify-center">
                      {sig.image_data ? (
                        <img
                          src={sig.image_data}
                          alt={`Sig ${index + 1}`}
                          className="max-w-full max-h-full object-contain rounded"
                        />
                      ) : (
                        <div className="text-stone-600 text-xs">
                          {index + 1}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {/* Matrix rows */}
              {matrix.map((row, rowIndex) => (
                <div key={rowIndex} className="flex mb-1">
                  {/* Row header */}
                  <div className="w-16 h-16 p-1">
                    <div className="w-full h-full bg-stone-100 rounded border border-stone-300 flex items-center justify-center">
                      {signatures[rowIndex]?.image_data ? (
                        <img
                          src={signatures[rowIndex].image_data}
                          alt={`Sig ${rowIndex + 1}`}
                          className="max-w-full max-h-full object-contain rounded"
                        />
                      ) : (
                        <div className="text-stone-600 text-xs">
                          {rowIndex + 1}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Matrix cells */}
                  {row.map((cell, colIndex) => (
                    <div
                      key={colIndex}
                      className={`w-16 h-16 p-1 cursor-pointer transition-all duration-200 ${
                        selectedCell === cell ? 'ring-2 ring-amber-600' : ''
                      }`}
                      onClick={() => setSelectedCell(selectedCell === cell ? null : cell)}
                      onMouseEnter={(e) => {
                        if (cell) {
                          const rect = e.currentTarget.getBoundingClientRect();
                          setShowTooltip({
                            x: rect.right + 10,
                            y: rect.top,
                            score: cell
                          });
                        }
                      }}
                      onMouseLeave={() => setShowTooltip(null)}
                    >
                      <div
                        className={`w-full h-full rounded border border-stone-300 flex items-center justify-center text-xs font-medium transition-all duration-300 ${
                          cell ? getSimilarityColor(cell.similarity) : 'bg-stone-200'
                        } ${cell ? getSimilarityIntensity(cell.similarity) : ''}`}
                        style={{
                          opacity: cell ? Math.max(0.3, cell.similarity) : 0.1
                        }}
                      >
                        {cell ? `${Math.round(cell.similarity * 100)}` : '—'}
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Tooltip */}
          {showTooltip && (
            <div
              className="fixed z-50 bg-white border border-stone-300 rounded-lg p-3 shadow-lg"
              style={{ left: showTooltip.x, top: showTooltip.y }}
            >
              <div className="text-stone-800 font-medium mb-2">
                Similarity: {formatFeatureScore(showTooltip.score.similarity)}%
              </div>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-blue-600">HOG:</span>
                  <span className="text-stone-800">{formatFeatureScore(showTooltip.score.features.hog)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-green-600">ResNet50:</span>
                  <span className="text-stone-800">{formatFeatureScore(showTooltip.score.features.resnet50)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-orange-600">VGG19:</span>
                  <span className="text-stone-800">{formatFeatureScore(showTooltip.score.features.vgg19)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-purple-600">ViT:</span>
                  <span className="text-stone-800">{formatFeatureScore(showTooltip.score.features.vit)}%</span>
                </div>
              </div>
            </div>
          )}

          {/* Selected Cell Details */}
          {selectedCell && (
            <div className="p-6 bg-stone-50/80 rounded-lg border border-green-200 shadow-inner">
              <div className="flex items-center space-x-3 mb-4">
                <Info className="w-5 h-5 text-green-700" />
                <h4 className="text-lg font-semibold text-stone-800">
                  Feature Comparison Breakdown
                </h4>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-green-700 mb-2">
                      {formatFeatureScore(selectedCell.similarity)}%
                    </div>
                    <div className="text-stone-600">Overall Similarity</div>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-blue-600 font-medium">HOG Features</span>
                      <span className="text-stone-800">{formatFeatureScore(selectedCell.features.hog)}%</span>
                    </div>
                    <div className="w-full bg-stone-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${selectedCell.features.hog * 100}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-green-600 font-medium">ResNet50</span>
                      <span className="text-stone-800">{formatFeatureScore(selectedCell.features.resnet50)}%</span>
                    </div>
                    <div className="w-full bg-stone-200 rounded-full h-2">
                      <div
                        className="bg-green-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${selectedCell.features.resnet50 * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-orange-600 font-medium">VGG19</span>
                      <span className="text-stone-800">{formatFeatureScore(selectedCell.features.vgg19)}%</span>
                    </div>
                    <div className="w-full bg-stone-200 rounded-full h-2">
                      <div
                        className="bg-orange-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${selectedCell.features.vgg19 * 100}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-purple-600 font-medium">Vision Transformer</span>
                      <span className="text-stone-800">{formatFeatureScore(selectedCell.features.vit)}%</span>
                    </div>
                    <div className="w-full bg-stone-200 rounded-full h-2">
                      <div
                        className="bg-purple-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${selectedCell.features.vit * 100}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  <div className="mt-4 p-3 bg-stone-100 rounded-lg">
                    <div className="text-sm text-stone-700">
                      <p className="mb-2"><strong>Feature Fusion:</strong></p>
                      <p className="text-xs text-stone-600">
                        This similarity score combines multiple deep learning features:
                        HOG for shape patterns, ResNet50 for general features, 
                        VGG19 for texture analysis, and Vision Transformer for contextual understanding.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Matrix Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-center">
              <div className="text-green-800 text-2xl font-bold">
                {similarityMatrix.filter(s => s.similarity >= 0.8).length}
              </div>
              <div className="text-green-700 text-sm">High Matches</div>
            </div>
            
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-center">
              <div className="text-amber-800 text-2xl font-bold">
                {similarityMatrix.filter(s => s.similarity >= 0.6 && s.similarity < 0.8).length}
              </div>
              <div className="text-amber-700 text-sm">Medium Matches</div>
            </div>
            
            <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 text-center">
              <div className="text-orange-800 text-2xl font-bold">
                {similarityMatrix.filter(s => s.similarity >= 0.4 && s.similarity < 0.6).length}
              </div>
              <div className="text-orange-700 text-sm">Low Matches</div>
            </div>
            
            <div className="bg-stone-50 border border-stone-200 rounded-lg p-4 text-center">
              <div className="text-stone-800 text-2xl font-bold">
                {similarityMatrix.length > 0 
                  ? (similarityMatrix.reduce((sum, s) => sum + s.similarity, 0) / similarityMatrix.length * 100).toFixed(0)
                  : 0}%
              </div>
              <div className="text-stone-700 text-sm">Avg Similarity</div>
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center py-12 text-stone-600">
          <Grid className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p>No similarity data available</p>
          <p className="text-sm mt-2">Complete signature detection to generate similarity matrix</p>
        </div>
      )}
    </div>
  );
};

export default SimilarityMatrix;