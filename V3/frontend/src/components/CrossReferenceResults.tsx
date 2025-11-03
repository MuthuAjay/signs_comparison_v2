import React from 'react';
import { CheckCircle, XCircle, FileText, Users, TrendingUp } from 'lucide-react';

interface TargetPDFResult {
  filename: string;
  total_signatures: number;
  signer_matches: Record<string, number>;
  signer_similarities: Record<string, {
    avg_similarity: number;
    avg_similarity_percentage: string;
    match_count: number;
  }>;
  processing_status: string;
  signature_details?: Array<{
    signature_id: string;
    page_number: number;
    matched_signer: string | null;
    similarity_score: number;
    similarity_percentage: string;
    confidence_level: string;
  }>;
}

interface CrossReferenceResultsProps {
  targetResults: Record<string, TargetPDFResult>;
  signerProfiles: Record<string, any>;
}

const CrossReferenceResults: React.FC<CrossReferenceResultsProps> = ({ 
  targetResults, 
  signerProfiles 
}) => {
  const targetPDFs = Object.entries(targetResults);
  const referenceSignerIDs = Object.keys(signerProfiles);

  if (targetPDFs.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-stone-200 p-8">
        <div className="text-center text-stone-500">
          <FileText className="w-12 h-12 mx-auto mb-4 text-stone-400" />
          <h3 className="text-lg font-medium mb-2">No Target PDFs Processed</h3>
          <p>Upload target PDFs to see cross-reference comparison results.</p>
        </div>
      </div>
    );
  }

  // Calculate summary statistics
  const totalTargetPDFs = targetPDFs.length;
  const successfulPDFs = targetPDFs.filter(([_, data]) => data.processing_status === 'success').length;
  const totalMatches = targetPDFs.reduce((sum, [_, data]) => {
    return sum + Object.values(data.signer_matches).reduce((a, b) => a + b, 0);
  }, 0);
  const averageMatchesPerPDF = totalTargetPDFs > 0 ? (totalMatches / totalTargetPDFs).toFixed(1) : '0';

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <FileText className="w-5 h-5 text-blue-600" />
            <span className="text-blue-700 text-sm font-medium">Target PDFs</span>
          </div>
          <div className="text-blue-800 text-2xl font-bold">{totalTargetPDFs}</div>
          <div className="text-blue-600 text-xs">{successfulPDFs} processed successfully</div>
        </div>
        
        <div className="bg-gradient-to-br from-green-50 to-green-100 border border-green-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Users className="w-5 h-5 text-green-600" />
            <span className="text-green-700 text-sm font-medium">Total Matches</span>
          </div>
          <div className="text-green-800 text-2xl font-bold">{totalMatches}</div>
          <div className="text-green-600 text-xs">Signatures matched</div>
        </div>
        
        <div className="bg-gradient-to-br from-purple-50 to-purple-100 border border-purple-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <TrendingUp className="w-5 h-5 text-purple-600" />
            <span className="text-purple-700 text-sm font-medium">Avg Matches</span>
          </div>
          <div className="text-purple-800 text-2xl font-bold">{averageMatchesPerPDF}</div>
          <div className="text-purple-600 text-xs">Per target PDF</div>
        </div>
        
        <div className="bg-gradient-to-br from-orange-50 to-orange-100 border border-orange-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <CheckCircle className="w-5 h-5 text-orange-600" />
            <span className="text-orange-700 text-sm font-medium">Success Rate</span>
          </div>
          <div className="text-orange-800 text-2xl font-bold">
            {totalTargetPDFs > 0 ? Math.round((successfulPDFs / totalTargetPDFs) * 100) : 0}%
          </div>
          <div className="text-orange-600 text-xs">Processing success</div>
        </div>
      </div>

      {/* Cross-Reference Results Table */}
      <div className="bg-white rounded-xl shadow-sm border border-stone-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-stone-200">
          <h3 className="text-lg font-semibold text-stone-800">📊 Cross-Reference Results</h3>
          <p className="text-stone-600 text-sm mt-1">
            Showing which reference signers were found in each target PDF
          </p>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-stone-50">
              <tr>
                <th className="text-left text-stone-600 font-medium py-3 px-4 border-b border-stone-200">
                  Target PDF
                </th>
                <th className="text-center text-stone-600 font-medium py-3 px-4 border-b border-stone-200">
                  Total Signatures
                </th>
                {referenceSignerIDs.map((signerId) => (
                  <th key={signerId} className="text-center text-stone-600 font-medium py-3 px-4 border-b border-stone-200">
                    <div className="space-y-1">
                      <div>Signer {signerId}</div>
                      <div className="text-xs text-stone-500">Count | Similarity</div>
                    </div>
                  </th>
                ))}
                <th className="text-center text-stone-600 font-medium py-3 px-4 border-b border-stone-200">
                  Total Matches
                </th>
                <th className="text-center text-stone-600 font-medium py-3 px-4 border-b border-stone-200">
                  Status
                </th>
              </tr>
            </thead>
            <tbody>
              {targetPDFs.map(([filename, data]) => {
                const totalMatches = Object.values(data.signer_matches).reduce((a, b) => a + b, 0);
                
                return (
                  <tr key={filename} className="border-b border-stone-100 hover:bg-stone-50">
                    <td className="py-4 px-4">
                      <div className="font-medium text-stone-800">{filename}</div>
                      <div className="text-stone-600 text-sm">
                        {data.signature_details?.length || 0} signatures analyzed
                      </div>
                    </td>
                    <td className="text-center py-4 px-4">
                      <span className="font-medium text-stone-800">{data.total_signatures}</span>
                    </td>
                    {referenceSignerIDs.map((signerId) => {
                      const matchCount = data.signer_matches[signerId] || 0;
                      const similarity = data.signer_similarities[signerId];
                      const similarityPercentage = similarity?.avg_similarity_percentage || '0.0%';
                      
                      return (
                        <td key={signerId} className="text-center py-4 px-4">
                          <div className="space-y-1">
                            <div className={`inline-flex items-center justify-center w-8 h-6 rounded-full text-xs font-medium ${
                              matchCount > 0 
                                ? 'bg-green-100 text-green-700 border border-green-200' 
                                : 'bg-gray-100 text-gray-500 border border-gray-200'
                            }`}>
                              {matchCount}
                            </div>
                            <div className={`text-xs ${
                              matchCount > 0 ? 'text-green-600' : 'text-gray-400'
                            }`}>
                              {similarityPercentage}
                            </div>
                          </div>
                        </td>
                      );
                    })}
                    <td className="text-center py-4 px-4">
                      <span className="font-bold text-blue-600">{totalMatches}</span>
                    </td>
                    <td className="text-center py-4 px-4">
                      <div className="flex items-center justify-center">
                        {data.processing_status === 'success' ? (
                          <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-700 border border-green-200">
                            <CheckCircle className="w-3 h-3 mr-1" />
                            Success
                          </span>
                        ) : data.processing_status === 'no_signatures' ? (
                          <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-700 border border-yellow-200">
                            <XCircle className="w-3 h-3 mr-1" />
                            No Signatures
                          </span>
                        ) : (
                          <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-700 border border-red-200">
                            <XCircle className="w-3 h-3 mr-1" />
                            Error
                          </span>
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

      {/* Detailed Signature Matches (Expandable sections) */}
      <div className="space-y-4">
        <h4 className="text-lg font-semibold text-stone-800">🔍 Detailed Signature Analysis</h4>
        {targetPDFs.map(([filename, data]) => {
          if (!data.signature_details || data.signature_details.length === 0) return null;
          
          return (
            <details key={filename} className="bg-white rounded-lg border border-stone-200">
              <summary className="cursor-pointer p-4 hover:bg-stone-50 font-medium text-stone-800">
                📄 {filename} - {data.signature_details.length} signature(s) detailed analysis
              </summary>
              <div className="px-4 pb-4">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-stone-200">
                        <th className="text-left py-2">Signature ID</th>
                        <th className="text-center py-2">Page</th>
                        <th className="text-center py-2">Matched Signer</th>
                        <th className="text-center py-2">Similarity</th>
                        <th className="text-center py-2">Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.signature_details.map((detail, index) => (
                        <tr key={index} className="border-b border-stone-100">
                          <td className="py-2 font-mono text-xs">{detail.signature_id}</td>
                          <td className="text-center py-2">{detail.page_number}</td>
                          <td className="text-center py-2">
                            {detail.matched_signer ? (
                              <span className="text-blue-600 font-medium">Signer {detail.matched_signer}</span>
                            ) : (
                              <span className="text-gray-400">No match</span>
                            )}
                          </td>
                          <td className="text-center py-2">{detail.similarity_percentage}</td>
                          <td className="text-center py-2">
                            <span className={`px-2 py-1 rounded text-xs ${
                              detail.confidence_level === 'High' 
                                ? 'bg-green-100 text-green-700'
                                : detail.confidence_level === 'Medium'
                                ? 'bg-yellow-100 text-yellow-700'
                                : 'bg-red-100 text-red-700'
                            }`}>
                              {detail.confidence_level}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </details>
          );
        })}
      </div>
    </div>
  );
};

export default CrossReferenceResults;