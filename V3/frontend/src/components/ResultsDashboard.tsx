import React from 'react';
import { Download, BarChart, Users, FileCheck, Eye } from 'lucide-react';
import { AnalysisResults } from '../types';

interface ResultsDashboardProps {
  results: AnalysisResults | null;
  onDownloadReport: () => void;
  loading?: boolean;
}

const ResultsDashboard: React.FC<ResultsDashboardProps> = ({
  results,
  onDownloadReport,
  loading = false
}) => {
  if (!results) {
    return (
      <div className="bg-white/90 backdrop-blur-md rounded-xl border border-stone-200 p-6 shadow-sm">
        <div className="text-center py-12 text-stone-600">
          <BarChart className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p>No analysis results yet</p>
          <p className="text-sm mt-2">Complete the analysis to see comprehensive results</p>
        </div>
      </div>
    );
  }

  const level1Clusters = results.clusters.filter(c => c.type === 'style');
  const level2Clusters = results.clusters.filter(c => c.type === 'individual');
  const highConfidenceSignatures = results.signatures.filter(s => s.confidence_score >= 0.8);
  const averageSimilarity = results.similarity_matrix.length > 0
    ? results.similarity_matrix.reduce((sum, s) => sum + s.similarity, 0) / results.similarity_matrix.length
    : 0;

  return (
    <div className="bg-white/90 backdrop-blur-md rounded-xl border border-stone-200 p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <BarChart className="w-6 h-6 text-amber-700" />
          <h3 className="text-xl font-semibold text-stone-800">Analysis Results</h3>
        </div>
        
        <button
          onClick={onDownloadReport}
          disabled={loading}
          className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700 text-white rounded-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed shadow-md"
        >
          <Download className="w-4 h-4" />
          <span>Download Report</span>
        </button>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <FileCheck className="w-5 h-5 text-blue-600" />
            <span className="text-blue-700 text-sm font-medium">Signatures</span>
          </div>
          <div className="text-blue-800 text-2xl font-bold">{results.signatures.length}</div>
          <div className="text-blue-600 text-xs">Total detected</div>
        </div>
        
        <div className="bg-gradient-to-br from-purple-50 to-purple-100 border border-purple-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Users className="w-5 h-5 text-purple-600" />
            <span className="text-purple-700 text-sm font-medium">Individuals</span>
          </div>
          <div className="text-purple-800 text-2xl font-bold">{level2Clusters.length}</div>
          <div className="text-purple-600 text-xs">Unique signers</div>
        </div>
        
        <div className="bg-gradient-to-br from-green-50 to-green-100 border border-green-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Eye className="w-5 h-5 text-green-600" />
            <span className="text-green-700 text-sm font-medium">High Quality</span>
          </div>
          <div className="text-green-800 text-2xl font-bold">{highConfidenceSignatures.length}</div>
          <div className="text-green-600 text-xs">≥80% confidence</div>
        </div>
        
        <div className="bg-gradient-to-br from-orange-50 to-orange-100 border border-orange-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <BarChart className="w-5 h-5 text-orange-600" />
            <span className="text-orange-700 text-sm font-medium">Avg Similarity</span>
          </div>
          <div className="text-orange-800 text-2xl font-bold">
            {(averageSimilarity * 100).toFixed(0)}%
          </div>
          <div className="text-orange-600 text-xs">Cross-document</div>
        </div>
      </div>

      {/* Detailed Results Table */}
      <div className="space-y-6">
        <h4 className="text-lg font-semibold text-stone-800">Signer Analysis</h4>
        
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b border-stone-300">
                <th className="text-left text-stone-600 font-medium py-3 px-4">Signer</th>
                <th className="text-left text-stone-600 font-medium py-3 px-4">Style Group</th>
                <th className="text-left text-stone-600 font-medium py-3 px-4">Signatures</th>
                <th className="text-left text-stone-600 font-medium py-3 px-4">Confidence</th>
                <th className="text-left text-stone-600 font-medium py-3 px-4">Documents</th>
                <th className="text-left text-stone-600 font-medium py-3 px-4">Preview</th>
              </tr>
            </thead>
            <tbody>
              {level2Clusters.map((signer) => {
                const styleGroup = level1Clusters.find(style => 
                  style.signatures.some(sig => 
                    signer.signatures.some(signerSig => signerSig.id === sig.id)
                  )
                );
                
                const uniquePages = new Set(signer.signatures.map(sig => sig.page_number));
                
                return (
                  <tr key={signer.id} className="border-b border-stone-200 hover:bg-stone-50">
                    <td className="py-4 px-4">
                      <div className="text-stone-800 font-medium">Signer {signer.name}</div>
                      <div className="text-stone-600 text-sm">{signer.id}</div>
                    </td>
                    <td className="py-4 px-4">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        styleGroup?.name.toLowerCase() === 'neat' 
                          ? 'bg-blue-100 text-blue-700 border border-blue-200'
                          : styleGroup?.name.toLowerCase() === 'messy'
                          ? 'bg-orange-100 text-orange-700 border border-orange-200'
                          : 'bg-purple-100 text-purple-700 border border-purple-200'
                      }`}>
                        {styleGroup?.name || 'Unknown'}
                      </span>
                    </td>
                    <td className="py-4 px-4">
                      <span className="text-stone-800 font-medium">{signer.signatures.length}</span>
                    </td>
                    <td className="py-4 px-4">
                      <div className="flex items-center space-x-2">
                        <div className="w-16 bg-stone-200 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              signer.confidence >= 0.8 
                                ? 'bg-green-600' 
                                : signer.confidence >= 0.6 
                                ? 'bg-amber-600' 
                                : 'bg-red-600'
                            }`}
                            style={{ width: `${signer.confidence * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-stone-800 text-sm font-medium">
                          {(signer.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td className="py-4 px-4">
                      <span className="text-stone-700">{uniquePages.size} pages</span>
                    </td>
                    <td className="py-4 px-4">
                      <div className="flex space-x-1">
                        {signer.signatures.slice(0, 3).map((sig, index) => (
                          <div key={sig.id} className="w-8 h-8 bg-stone-100 rounded border border-stone-300 flex items-center justify-center">
                            {sig.image_data ? (
                              <img
                                src={sig.image_data}
                                alt={`Sig ${index + 1}`}
                                className="max-w-full max-h-full object-contain rounded"
                              />
                            ) : (
                              <div className="w-2 h-2 bg-stone-400 rounded-full"></div>
                            )}
                          </div>
                        ))}
                        {signer.signatures.length > 3 && (
                          <div className="w-8 h-8 bg-stone-100 rounded border border-stone-300 flex items-center justify-center">
                            <span className="text-stone-600 text-xs">+{signer.signatures.length - 3}</span>
                          </div>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-stone-50/80 rounded-lg p-4 border border-stone-200 shadow-inner">
          <h5 className="text-stone-800 font-semibold mb-3">Style Distribution</h5>
          <div className="space-y-2">
            {level1Clusters.map((cluster) => (
              <div key={cluster.id} className="flex items-center justify-between">
                <span className="text-stone-700 capitalize">{cluster.name}</span>
                <div className="flex items-center space-x-2">
                  <div className="w-20 bg-stone-200 rounded-full h-2">
                    <div
                      className="bg-amber-600 h-2 rounded-full"
                      style={{ width: `${(cluster.signatures.length / results.signatures.length) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-stone-600 text-sm">{cluster.signatures.length}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        <div className="bg-stone-50/80 rounded-lg p-4 border border-stone-200 shadow-inner">
          <h5 className="text-stone-800 font-semibold mb-3">Quality Breakdown</h5>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-green-700">High Quality (≥80%)</span>
              <span className="text-green-800 font-medium">{highConfidenceSignatures.length}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-amber-700">Medium Quality (60-79%)</span>
              <span className="text-amber-800 font-medium">
                {results.signatures.filter(s => s.confidence_score >= 0.6 && s.confidence_score < 0.8).length}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-red-700">Low Quality (&lt;60%)</span>
              <span className="text-red-800 font-medium">
                {results.signatures.filter(s => s.confidence_score < 0.6).length}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsDashboard;