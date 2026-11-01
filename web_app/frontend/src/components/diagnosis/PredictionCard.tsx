import React from 'react';
import type { PredictionResponse } from '../../types/prediction';
import { Clock } from 'lucide-react';

interface Props {
  result: PredictionResponse;
  image: File | null;
}

const PredictionCard: React.FC<Props> = ({ result, image }) => {
  const imageUrl = image ? URL.createObjectURL(image) : undefined;
  
  // Format confidence to percentage
  const confidencePercent = (result.confidence * 100).toFixed(1);
  
  // Color coding based on confidence
  let confidenceColor = "text-green-600 bg-green-50 border-green-100";
  if (result.confidence < 0.6) {
    confidenceColor = "text-red-600 bg-red-50 border-red-100";
  } else if (result.confidence < 0.8) {
    confidenceColor = "text-yellow-600 bg-yellow-50 border-yellow-100";
  }

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
      <div className="h-48 w-full bg-gray-100 relative">
        {imageUrl && (
          <img src={imageUrl} alt="Uploaded" className="w-full h-full object-cover" />
        )}
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
        <div className="absolute bottom-4 left-4 right-4">
          <p className="text-white/80 text-xs font-medium uppercase tracking-wider mb-1">Top Prediction</p>
          <h2 className="text-2xl font-bold text-white capitalize truncate">{result.prediction}</h2>
        </div>
      </div>
      
      <div className="p-6">
        <div className="flex justify-between items-center mb-6">
          <span className="text-sm font-medium text-gray-500">Confidence Score</span>
          <span className={`px-3 py-1 rounded-full text-sm font-bold border ${confidenceColor}`}>
            {confidencePercent}%
          </span>
        </div>
        
        {/* Simple Progress Bar */}
        <div className="w-full bg-slate-100 rounded-full h-2.5 mb-6 overflow-hidden">
          <div 
            className="bg-gradient-to-r from-blue-600 to-indigo-600 h-2.5 rounded-full transition-all duration-1000 ease-out" 
            style={{ width: `${confidencePercent}%` }}
          ></div>
        </div>
        
        <div className="flex items-center justify-between text-xs text-gray-400 pt-4 border-t border-gray-50">
          <div className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            Inference Time
          </div>
          <span>{result.latency_ms.toFixed(0)} ms</span>
        </div>
      </div>
    </div>
  );
};

export default PredictionCard;
