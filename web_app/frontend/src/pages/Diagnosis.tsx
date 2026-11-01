import React from 'react';
import { useAppStore } from '../app/store';
import ImageDropzone from '../components/upload/ImageDropzone';
import PredictionCard from '../components/diagnosis/PredictionCard';
import TopPredictions from '../components/diagnosis/TopPredictions';
import ClassProbabilityChart from '../components/diagnosis/ClassProbabilityChart';
import ClinicalSummary from '../components/diagnosis/ClinicalSummary';

const Diagnosis: React.FC = () => {
  const { predictionResult, resetDiagnosis, selectedImage } = useAppStore();

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <div className="flex justify-between items-end mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Diagnosis</h1>
          <p className="text-gray-500 mt-2">Upload a dermatological image for AI analysis.</p>
        </div>
        {predictionResult && (
          <button 
            onClick={resetDiagnosis}
            className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-800 rounded-lg transition-colors text-sm font-medium"
          >
            New Analysis
          </button>
        )}
      </div>

      {!predictionResult ? (
        <div className="max-w-3xl mx-auto">
           <ImageDropzone />
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-1 space-y-6">
            <PredictionCard result={predictionResult} image={selectedImage} />
            <TopPredictions predictions={predictionResult.top_3_predictions} />
          </div>
          
          <div className="lg:col-span-2 space-y-6">
            <ClinicalSummary report={predictionResult.report} />
            <ClassProbabilityChart probabilities={predictionResult.class_probabilities} />
            {predictionResult.heatmap_url && (
              <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Explainability Heatmap (Grad-CAM)</h3>
                <img 
                  src={predictionResult.heatmap_url.startsWith('http') ? predictionResult.heatmap_url : `http://localhost:8000${predictionResult.heatmap_url}`} 
                  alt="GradCAM Heatmap" 
                  className="rounded-lg max-h-[400px] object-contain w-full bg-black/5"
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Diagnosis;
