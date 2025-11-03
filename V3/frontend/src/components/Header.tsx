import React from 'react';
import { FileSearch, Wifi, WifiOff } from 'lucide-react';

interface HeaderProps {
  isConnected: boolean;
  onConnectionCheck: () => void;
}

const Header: React.FC<HeaderProps> = ({ isConnected, onConnectionCheck }) => {
  return (
    <header className="relative bg-gradient-to-r from-stone-100/90 to-amber-50/90 backdrop-blur-md border-b border-amber-200/40 shadow-sm">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="relative">
              <FileSearch className="w-10 h-10 text-amber-700" />
              <div className="absolute -inset-1 bg-gradient-to-r from-amber-400 to-orange-400 rounded-full blur opacity-20 animate-pulse"></div>
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-stone-800 via-amber-700 to-orange-600 bg-clip-text text-transparent">
                PDF Signature Detection & Clustering
              </h1>
              <p className="text-stone-600 text-sm mt-1">
                Advanced AI-powered signature analysis with two-level clustering algorithms
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <button
              onClick={onConnectionCheck}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg border transition-all duration-300 shadow-sm ${
                isConnected
                  ? 'bg-green-50 border-green-200 text-green-700 hover:bg-green-100'
                  : 'bg-red-50 border-red-200 text-red-700 hover:bg-red-100'
              }`}
            >
              {isConnected ? (
                <Wifi className="w-4 h-4" />
              ) : (
                <WifiOff className="w-4 h-4" />
              )}
              <span className="text-sm font-medium">
                {isConnected ? 'API Connected' : 'API Disconnected'}
              </span>
            </button>
            
            <div className="hidden md:flex items-center space-x-2 text-xs text-stone-600">
              <span>Endpoint:</span>
              <code className="bg-stone-200/70 px-2 py-1 rounded font-mono text-stone-800">
                localhost:5000/api
              </code>
            </div>
          </div>
        </div>
      </div>
      
      {/* Animated background gradient */}
      <div className="absolute inset-0 bg-gradient-to-r from-amber-100/20 via-stone-100/20 to-orange-100/20 animate-gradient-x"></div>
    </header>
  );
};

export default Header;