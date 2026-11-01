import React from 'react';
import type { ClassProbability } from '../../types/prediction';

interface Props {
  predictions: ClassProbability[];
}

const TopPredictions: React.FC<Props> = ({ predictions }) => {
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Differential Diagnosis</h3>
      <div className="space-y-4">
        {predictions.slice(1).map((pred, idx) => (
          <div key={idx} className="flex justify-between items-center">
            <span className="text-sm font-medium text-gray-700 capitalize">{pred.class_name}</span>
            <span className="text-sm font-bold text-gray-500">
              {(pred.probability * 100).toFixed(1)}%
            </span>
          </div>
        ))}
        {predictions.length <= 1 && (
           <p className="text-sm text-gray-400 italic">No alternative diagnoses available.</p>
        )}
      </div>
    </div>
  );
};

export default TopPredictions;
