import React, { useState, useEffect } from 'react';
import { Users, Eye, BarChart3, Target } from 'lucide-react';
import { ClusterGroup, SignatureDetection } from '../types';

interface ClusteringAnimationProps {
  clusters: ClusterGroup[];
  signatures: SignatureDetection[];
  loading?: boolean;
}

const ClusteringAnimation: React.FC<ClusteringAnimationProps> = ({
  clusters,
  signatures,
  loading = false
}) => {
  const [animationStep, setAnimationStep] = useState<'initial' | 'level1' | 'level2'>('initial');
  const [selectedCluster, setSelectedCluster] = useState<ClusterGroup | null>(null);

  useEffect(() => {
    if (clusters.length > 0 && !loading) {
      const timer1 = setTimeout(() => setAnimationStep('level1'), 500);
      const timer2 = setTimeout(() => setAnimationStep('level2'), 2000);
      
      return () => {
        clearTimeout(timer1);
        clearTimeout(timer2);
      };
    }
  }, [clusters.length, loading]);

  const level1Clusters = clusters.filter(c => c.type === 'style');
  const level2Clusters = clusters.filter(c => c.type === 'individual');

  const getClusterColor = (cluster: ClusterGroup) => {
    const colors = {
      'neat': 'from-blue-500 to-cyan-500',
      'messy': 'from-orange-500 to-red-500',
      'artistic': 'from-purple-500 to-pink-500',
      'default': 'from-slate-500 to-slate-600'
    };
    
    const style = cluster.name.toLowerCase();
    return colors[style as keyof typeof colors] || colors.default;
  };

  const getBorderColor = (cluster: ClusterGroup) => {
    const colors = {
      'neat': 'border-cyan-500',
      'messy': 'border-orange-500',
      'artistic': 'border-purple-500',
      'default': 'border-slate-500'
    };
    
    const style = cluster.name.toLowerCase();
    return colors[style as keyof typeof colors] || colors.default;
  };

  return (
    <div className="bg-white/90 backdrop-blur-md rounded-xl border border-stone-200 p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Users className="w-6 h-6 text-amber-700" />
          <h3 className="text-xl font-semibold text-stone-800">Two-Level Clustering</h3>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 text-sm text-stone-600">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            <span>Level 1: Style</span>
          </div>
          <div className="flex items-center space-x-2 text-sm text-stone-600">
            <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
            <span>Level 2: Individual</span>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-amber-600"></div>
          <span className="ml-4 text-stone-600">Analyzing signature patterns...</span>
        </div>
      ) : clusters.length > 0 ? (
        <div className="space-y-6">
          {/* Clustering Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gradient-to-r from-blue-50 to-blue-100 border border-blue-200 rounded-lg p-4">
              <div className="flex items-center space-x-2">
                <Target className="w-5 h-5 text-blue-400" />
                <span className="text-blue-700 text-sm font-medium">Style Groups</span>
              </div>
              <div className="text-blue-800 text-2xl font-bold mt-1">
                {level1Clusters.length}
              </div>
            </div>
            
            <div className="bg-gradient-to-r from-purple-50 to-purple-100 border border-purple-200 rounded-lg p-4">
              <div className="flex items-center space-x-2">
                <Users className="w-5 h-5 text-purple-400" />
                <span className="text-purple-700 text-sm font-medium">Individuals</span>
              </div>
              <div className="text-purple-800 text-2xl font-bold mt-1">
                {level2Clusters.length}
              </div>
            </div>
            
            <div className="bg-gradient-to-r from-green-50 to-green-100 border border-green-200 rounded-lg p-4">
              <div className="flex items-center space-x-2">
                <BarChart3 className="w-5 h-5 text-green-400" />
                <span className="text-green-700 text-sm font-medium">Avg Confidence</span>
              </div>
              <div className="text-green-800 text-2xl font-bold mt-1">
                {clusters.length > 0 
                  ? (clusters.reduce((sum, c) => sum + c.confidence, 0) / clusters.length * 100).toFixed(0)
                  : 0}%
              </div>
            </div>
            
            <div className="bg-gradient-to-r from-amber-50 to-amber-100 border border-amber-200 rounded-lg p-4">
              <div className="flex items-center space-x-2">
                <Eye className="w-5 h-5 text-yellow-400" />
                <span className="text-amber-700 text-sm font-medium">Total Signatures</span>
              </div>
              <div className="text-amber-800 text-2xl font-bold mt-1">
                {signatures.length}
              </div>
            </div>
          </div>

          {/* Level 1 Clustering - Style Groups */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-1 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full"></div>
              <h4 className="text-lg font-semibold text-stone-800">Level 1: Style-Based Clustering</h4>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {level1Clusters.map((cluster, index) => (
                <div
                  key={cluster.id}
                  className={`relative group cursor-pointer transition-all duration-500 ${
                    animationStep === 'initial' 
                      ? 'opacity-0 translate-y-4' 
                      : 'opacity-100 translate-y-0'
                  } ${
                    selectedCluster?.id === cluster.id
                      ? 'scale-105 ring-2 ring-amber-500'
                      : 'hover:scale-102'
                  }`}
                  style={{ transitionDelay: `${index * 200}ms` }}
                  onClick={() => setSelectedCluster(
                    selectedCluster?.id === cluster.id ? null : cluster
                  )}
                >
                  <div className={`bg-gradient-to-br ${getClusterColor(cluster)}/10 backdrop-blur-sm rounded-lg p-4 border ${getBorderColor(cluster)}/30 hover:border-opacity-60 transition-all duration-300`}>
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-stone-800 font-semibold capitalize">{cluster.name} Style</span>
                      <span className="text-xs px-2 py-1 rounded-full bg-stone-100 text-stone-700">
                        {cluster.signatures.length} sigs
                      </span>
                    </div>
                    
                    <div className="grid grid-cols-4 gap-1 mb-3">
                      {cluster.signatures.slice(0, 8).map((sig, sigIndex) => (
                        <div
                          key={sig.id}
                          className="aspect-square bg-stone-100 rounded border border-stone-300 flex items-center justify-center"
                        >
                          {sig.image_data ? (
                            <img
                              src={sig.image_data}
                              alt={`Signature ${sig.id}`}
                              className="max-w-full max-h-full object-contain rounded"
                            />
                          ) : (
                            <div className="w-2 h-2 bg-slate-500 rounded-full"></div>
                          )}
                        </div>
                      ))}
                    </div>
                    
                    <div className="flex justify-between text-sm">
                      <span className="text-stone-600">Confidence:</span>
                      <span className="text-stone-800 font-medium">
                        {(cluster.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Level 2 Clustering - Individual Separation */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-1 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"></div>
              <h4 className="text-lg font-semibold text-stone-800">Level 2: Individual Separation</h4>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {level2Clusters.map((cluster, index) => (
                <div
                  key={cluster.id}
                  className={`relative group cursor-pointer transition-all duration-700 ${
                    animationStep !== 'level2' 
                      ? 'opacity-0 scale-50 translate-y-8' 
                      : 'opacity-100 scale-100 translate-y-0'
                  } ${
                    selectedCluster?.id === cluster.id
                      ? 'ring-2 ring-purple-600'
                      : 'hover:ring-1 hover:ring-purple-400/50'
                  }`}
                  style={{ transitionDelay: `${index * 100}ms` }}
                  onClick={() => setSelectedCluster(
                    selectedCluster?.id === cluster.id ? null : cluster
                  )}
                >
                  <div className="bg-white/80 rounded-lg p-3 border border-purple-200 shadow-sm hover:border-purple-400/50 transition-all duration-300">
                    <div className="aspect-square bg-stone-100 rounded mb-2 flex items-center justify-center">
                      {cluster.signatures[0]?.image_data ? (
                        <img
                          src={cluster.signatures[0].image_data}
                          alt={`Individual ${cluster.name}`}
                          className="max-w-full max-h-full object-contain rounded"
                        />
                      ) : (
                        <div className="w-6 h-6 bg-purple-400 rounded-full opacity-50"></div>
                      )}
                    </div>
                    
                    <div className="text-center">
                      <div className="text-stone-800 text-xs font-medium mb-1">
                        Signer {cluster.name}
                      </div>
                      <div className="text-purple-700 text-xs">
                        {cluster.signatures.length} signature{cluster.signatures.length !== 1 ? 's' : ''}
                      </div>
                      <div className="text-stone-600 text-xs mt-1">
                        {(cluster.confidence * 100).toFixed(0)}%
                      </div>
                    </div>

                    {/* Animated clustering indicator */}
                    <div className="absolute -inset-1 bg-gradient-to-r from-purple-400/20 to-pink-400/20 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Selected Cluster Details */}
          {selectedCluster && (
            <div className="p-6 bg-stone-50/80 rounded-lg border border-amber-200 shadow-inner">
              <div className="flex items-center space-x-3 mb-4">
                <div className={`w-4 h-4 rounded-full ${selectedCluster.type === 'style' ? 'bg-blue-400' : 'bg-purple-400'}`}></div>
                <h4 className="text-lg font-semibold text-stone-800">
                  {selectedCluster.type === 'style' ? 'Style Group' : 'Individual Signer'}: {selectedCluster.name}
                </h4>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-stone-600">Type:</span>
                    <span className="text-stone-800 font-medium capitalize">{selectedCluster.type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-stone-600">Signatures:</span>
                    <span className="text-stone-800 font-medium">{selectedCluster.signatures.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-stone-600">Confidence:</span>
                    <span className="text-stone-800 font-medium">{(selectedCluster.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>
                
                <div className="grid grid-cols-4 gap-2">
                  {selectedCluster.signatures.map((sig, index) => (
                    <div key={sig.id} className="aspect-square bg-stone-100 rounded border border-stone-300 flex items-center justify-center">
                      {sig.image_data ? (
                        <img
                          src={sig.image_data}
                          alt={`Signature ${sig.id}`}
                          className="max-w-full max-h-full object-contain rounded"
                        />
                      ) : (
                        <div className="w-2 h-2 bg-slate-500 rounded-full"></div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="text-center py-12 text-stone-600">
          <Users className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p>No clusters generated yet</p>
          <p className="text-sm mt-2">Complete signature detection to see clustering results</p>
        </div>
      )}
    </div>
  );
};

export default ClusteringAnimation;