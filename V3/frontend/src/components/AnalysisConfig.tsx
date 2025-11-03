import React from 'react';
import { Settings, Sliders, Brain, FileType } from 'lucide-react';
import { AnalysisConfig } from '../types';

interface AnalysisConfigProps {
  config: AnalysisConfig;
  onChange: (config: AnalysisConfig) => void;
  disabled?: boolean;
}

const AnalysisConfiguration: React.FC<AnalysisConfigProps> = ({
  config,
  onChange,
  disabled = false
}) => {
  const updateConfig = (key: keyof AnalysisConfig, value: any) => {
    const newConfig = { ...config, [key]: value };
    if (key === 'vgg_weight') {
      newConfig.vit_weight = 1 - value;
    }
    onChange(newConfig);
  };

  return (
    <div className="bg-white/90 backdrop-blur-md rounded-xl border border-stone-200 p-6 shadow-sm">
      <div className="flex items-center space-x-3 mb-6">
        <Settings className="w-6 h-6 text-amber-700" />
        <h3 className="text-xl font-semibold text-stone-800">Analysis Configuration</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Similarity Threshold */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <Sliders className="w-4 h-4 text-orange-600" />
            <label className="text-stone-800 font-medium">Similarity Threshold</label>
          </div>
          <div className="space-y-2">
            <input
              type="range"
              min="0.5"
              max="0.9"
              step="0.05"
              value={config.similarity_threshold}
              onChange={(e) => updateConfig('similarity_threshold', parseFloat(e.target.value))}
              disabled={disabled}
              className="w-full h-2 bg-stone-200 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between text-sm text-stone-600">
              <span>0.5 (Loose)</span>
              <span className="text-amber-700 font-semibold">{config.similarity_threshold}</span>
              <span>0.9 (Strict)</span>
            </div>
          </div>
          <p className="text-xs text-stone-600">
            Higher values require more similar signatures to match
          </p>
        </div>

        {/* Model Path */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <FileType className="w-4 h-4 text-green-600" />
            <label className="text-stone-800 font-medium">Model Path</label>
          </div>
          <input
            type="text"
            value={config.model_path}
            onChange={(e) => updateConfig('model_path', e.target.value)}
            disabled={disabled}
            placeholder="path/to/model.pkl"
            className="w-full px-4 py-2 bg-stone-50 border border-stone-300 rounded-lg text-stone-800 placeholder-stone-500 focus:border-amber-500 focus:ring-2 focus:ring-amber-500/20 transition-colors duration-300 shadow-inner"
          />
          <p className="text-xs text-stone-600">
            Path to the trained signature detection model
          </p>
        </div>

        {/* VGG Weight */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <Brain className="w-4 h-4 text-blue-600" />
            <label className="text-stone-800 font-medium">VGG19 Weight</label>
          </div>
          <div className="space-y-2">
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={config.vgg_weight}
              onChange={(e) => updateConfig('vgg_weight', parseFloat(e.target.value))}
              disabled={disabled}
              className="w-full h-2 bg-stone-200 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between text-sm text-stone-600">
              <span>0.0</span>
              <span className="text-blue-700 font-semibold">{config.vgg_weight}</span>
              <span>1.0</span>
            </div>
          </div>
          <p className="text-xs text-stone-600">
            VGG19: {config.vgg_weight}, ViT: {config.vit_weight.toFixed(1)}
          </p>
        </div>

        {/* Feature Extraction Info */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <Brain className="w-4 h-4 text-purple-600" />
            <label className="text-stone-800 font-medium">Feature Extraction</label>
          </div>
          <div className="bg-stone-50/80 border border-stone-200 rounded-lg p-3 space-y-2 shadow-inner">
            <div className="flex justify-between text-sm">
              <span className="text-stone-600">VGG19</span>
              <span className="text-blue-700 font-medium">Weight: {config.vgg_weight}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-stone-600">Vision Transformer</span>
              <span className="text-purple-700 font-medium">Weight: {config.vit_weight.toFixed(1)}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6 p-4 bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg border border-amber-200 shadow-inner">
        <h4 className="text-stone-800 font-semibold mb-2">Algorithm Overview</h4>
        <div className="text-sm text-stone-700 space-y-1">
          <p>• <strong className="text-stone-800">Level 1 Clustering:</strong> Groups signatures by style (neat, messy, artistic)</p>
          <p>• <strong className="text-stone-800">Level 2 Clustering:</strong> Separates individuals within each style group</p>
          <p>• <strong className="text-stone-800">Multi-Feature Fusion:</strong> Combines VGG19, and ViT features</p>
        </div>
      </div>
    </div>
  );
};

export default AnalysisConfiguration;