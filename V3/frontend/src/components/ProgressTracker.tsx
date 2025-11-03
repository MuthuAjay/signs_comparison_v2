import React from 'react';
import { Clock, CheckCircle, AlertCircle, Loader } from 'lucide-react';
import { AnalysisStatus } from '../types';

interface ProgressTrackerProps {
  status: AnalysisStatus | null;
  className?: string;
}

const ProgressTracker: React.FC<ProgressTrackerProps> = ({ status, className = '' }) => {
  if (!status) return null;

  const getStatusIcon = () => {
    switch (status.status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-600" />;
      case 'processing':
        return <Loader className="w-5 h-5 text-amber-700 animate-spin" />;
      default:
        return <Clock className="w-5 h-5 text-amber-600" />;
    }
  };

  const getStatusColor = () => {
    switch (status.status) {
      case 'completed':
        return 'from-green-500 to-emerald-500';
      case 'error':
        return 'from-red-500 to-rose-500';
      case 'processing':
        return 'from-amber-500 to-orange-500';
      default:
        return 'from-amber-500 to-orange-500';
    }
  };

  const steps = [
    'Initializing analysis...',
    'Extracting signatures from PDFs...',
    'Level 1 Clustering (Style Groups)...',
    'Level 2 Clustering (Individuals)...',
    'Computing similarity matrix...',
    'Generating report...',
    'Analysis complete!'
  ];

  const currentStepIndex = steps.findIndex(step => 
    status.current_step.toLowerCase().includes(step.toLowerCase().split(' ')[0])
  );

  return (
    <div className={`bg-white/90 backdrop-blur-md rounded-xl border border-stone-200 p-6 shadow-sm ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          {getStatusIcon()}
          <h3 className="text-xl font-semibold text-stone-800">Analysis Progress</h3>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-stone-800">{status.progress}%</div>
          <div className="text-sm text-stone-600 capitalize">{status.status}</div>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-6">
        <div className="relative w-full bg-stone-200 rounded-full h-3 overflow-hidden">
          <div
            className={`absolute left-0 top-0 h-full bg-gradient-to-r ${getStatusColor()} transition-all duration-500 ease-out`}
            style={{ width: `${status.progress}%` }}
          >
            <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
          </div>
        </div>
      </div>

      {/* Current Step */}
      <div className="mb-6 p-4 bg-stone-50/80 rounded-lg border border-amber-200 shadow-inner">
        <div className="flex items-center space-x-3">
          <div className="w-2 h-2 bg-amber-600 rounded-full animate-pulse"></div>
          <span className="text-stone-800 font-medium">Current Step:</span>
        </div>
        <p className="text-amber-700 mt-2">{status.current_step}</p>
      </div>

      {/* Step Timeline */}
      <div className="space-y-3">
        <h4 className="text-stone-800 font-medium mb-3">Analysis Pipeline</h4>
        {steps.map((step, index) => {
          const isCompleted = index < currentStepIndex;
          const isCurrent = index === currentStepIndex;
          const isPending = index > currentStepIndex;

          return (
            <div
              key={index}
              className={`flex items-center space-x-3 p-2 rounded-lg transition-all duration-300 ${
                isCurrent
                  ? 'bg-amber-50 border border-amber-200'
                  : isCompleted
                  ? 'bg-green-50 border border-green-200'
                  : 'bg-stone-50 border border-stone-200'
              }`}
            >
              <div
                className={`w-3 h-3 rounded-full transition-colors duration-300 ${
                  isCompleted
                    ? 'bg-green-600'
                    : isCurrent
                    ? 'bg-amber-600 animate-pulse'
                    : 'bg-stone-400'
                }`}
              ></div>
              <span
                className={`text-sm transition-colors duration-300 ${
                  isCompleted
                    ? 'text-green-700'
                    : isCurrent
                    ? 'text-amber-700'
                    : 'text-stone-600'
                }`}
              >
                {step}
              </span>
              {isCompleted && (
                <CheckCircle className="w-4 h-4 text-green-600 ml-auto" />
              )}
            </div>
          );
        })}
      </div>

      {/* Error Message */}
      {status.error && (
        <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-5 h-5 text-red-600" />
            <span className="text-red-700 font-medium">Error</span>
          </div>
          <p className="text-red-700 mt-2 text-sm">{status.error}</p>
        </div>
      )}
    </div>
  );
};

export default ProgressTracker;