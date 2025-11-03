import React, { useCallback, useState } from 'react';
import { Upload, FileText, X, CheckCircle } from 'lucide-react';

interface FileUploadProps {
  onReferenceUpload: (file: File) => void;
  onTargetUpload: (files: File[]) => void;
  referenceFile: File | null;
  targetFiles: File[];
  loading: boolean;
}

const FileUpload: React.FC<FileUploadProps> = ({
  onReferenceUpload,
  onTargetUpload,
  referenceFile,
  targetFiles,
  loading
}) => {
  const [dragOver, setDragOver] = useState<'reference' | 'target' | null>(null);

  const handleDrag = useCallback((e: React.DragEvent, type: 'reference' | 'target' | null) => {
    e.preventDefault();
    setDragOver(type);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent, type: 'reference' | 'target') => {
    e.preventDefault();
    setDragOver(null);
    
    const files = Array.from(e.dataTransfer.files).filter(file => 
      file.type === 'application/pdf'
    );
    
    if (files.length === 0) return;
    
    if (type === 'reference') {
      onReferenceUpload(files[0]);
    } else {
      onTargetUpload(files);
    }
  }, [onReferenceUpload, onTargetUpload]);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="bg-white/90 backdrop-blur-md border border-stone-200 rounded-xl p-6 shadow-sm">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Reference File Upload */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-stone-800 flex items-center space-x-2">
            <FileText className="w-5 h-5 text-amber-700" />
            <span>Reference Document</span>
          </h3>
          
          <div
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 shadow-inner ${
              dragOver === 'reference'
                ? 'border-amber-500 bg-amber-50/80 scale-105 shadow-amber-100'
                : referenceFile
                ? 'border-green-500 bg-green-50/80 shadow-green-100'
                : 'border-stone-300 hover:border-amber-400 hover:bg-amber-50/30 bg-stone-50/50'
            }`}
            onDragOver={(e) => handleDrag(e, 'reference')}
            onDragLeave={(e) => handleDrag(e, null)}
            onDrop={(e) => handleDrop(e, 'reference')}
          >
            {referenceFile ? (
              <div className="space-y-3">
                <CheckCircle className="w-12 h-12 text-green-600 mx-auto" />
                <div>
                  <p className="text-green-700 font-medium">{referenceFile.name}</p>
                  <p className="text-stone-600 text-sm">{formatFileSize(referenceFile.size)}</p>
                </div>
                <button
                  onClick={() => onReferenceUpload(referenceFile)}
                  className="text-red-600 hover:text-red-700 text-sm underline transition-colors duration-200"
                  disabled={loading}
                >
                  Remove
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                <Upload className="w-12 h-12 text-stone-500 mx-auto" />
                <div>
                  <p className="text-stone-800 font-medium">Drop your reference PDF here</p>
                  <p className="text-stone-600 text-sm mt-1">
                    Contains signatures to compare against
                  </p>
                </div>
                <input
                  type="file"
                  accept=".pdf"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) onReferenceUpload(file);
                  }}
                  className="hidden"
                  id="reference-upload"
                  disabled={loading}
                />
                <label
                  htmlFor="reference-upload"
                  className="inline-block px-4 py-2 bg-amber-100 hover:bg-amber-200 text-amber-800 border border-amber-300 rounded-lg cursor-pointer transition-all duration-300 shadow-sm hover:shadow-md font-medium"
                >
                  Choose File
                </label>
              </div>
            )}
          </div>
        </div>

        {/* Target Files Upload */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-stone-800 flex items-center space-x-2">
            <FileText className="w-5 h-5 text-orange-700" />
            <span>Target Documents</span>
          </h3>
          
          <div
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 min-h-[200px] shadow-inner ${
              dragOver === 'target'
                ? 'border-orange-500 bg-orange-50/80 scale-105 shadow-orange-100'
                : targetFiles.length > 0
                ? 'border-green-500 bg-green-50/80 shadow-green-100'
                : 'border-stone-300 hover:border-orange-400 hover:bg-orange-50/30 bg-stone-50/50'
            }`}
            onDragOver={(e) => handleDrag(e, 'target')}
            onDragLeave={(e) => handleDrag(e, null)}
            onDrop={(e) => handleDrop(e, 'target')}
          >
            {targetFiles.length > 0 ? (
              <div className="space-y-4">
                <CheckCircle className="w-12 h-12 text-green-600 mx-auto" />
                <div className="space-y-2 max-h-32 overflow-y-auto scrollbar-thin">
                  {targetFiles.map((file, index) => (
                    <div key={index} className="flex items-center justify-between bg-white/70 border border-stone-200 rounded-lg p-3 shadow-sm">
                      <div className="text-left">
                        <p className="text-green-700 font-medium text-sm">{file.name}</p>
                        <p className="text-stone-600 text-xs">{formatFileSize(file.size)}</p>
                      </div>
                      <button
                        onClick={() => {
                          const newFiles = targetFiles.filter((_, i) => i !== index);
                          onTargetUpload(newFiles);
                        }}
                        className="text-red-600 hover:text-red-700 p-1 hover:bg-red-50 rounded-full transition-colors duration-200"
                        disabled={loading}
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                </div>
                <p className="text-stone-600 text-sm font-medium">
                  {targetFiles.length} file{targetFiles.length !== 1 ? 's' : ''} selected
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                <Upload className="w-12 h-12 text-stone-500 mx-auto" />
                <div>
                  <p className="text-stone-800 font-medium">Drop target PDFs here</p>
                  <p className="text-stone-600 text-sm mt-1">
                    Documents to search for signatures
                  </p>
                </div>
                <input
                  type="file"
                  accept=".pdf"
                  multiple
                  onChange={(e) => {
                    const files = Array.from(e.target.files || []);
                    if (files.length > 0) onTargetUpload(files);
                  }}
                  className="hidden"
                  id="target-upload"
                  disabled={loading}
                />
                <label
                  htmlFor="target-upload"
                  className="inline-block px-4 py-2 bg-orange-100 hover:bg-orange-200 text-orange-800 border border-orange-300 rounded-lg cursor-pointer transition-all duration-300 shadow-sm hover:shadow-md font-medium"
                >
                  Choose Files
                </label>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default FileUpload;