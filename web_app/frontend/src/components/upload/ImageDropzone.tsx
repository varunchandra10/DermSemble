import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  UploadCloud, 
  FileImage, 
  X, 
  Loader2, 
  Sparkles, 
  ArrowRight,
  RotateCcw
} from 'lucide-react';
import { useAppStore } from '../../app/store';
import { usePrediction } from '../../hooks/usePrediction';

const ImageDropzone: React.FC = () => {
  const { selectedImage, imagePreviewUrl, setSelectedImage } = useAppStore();
  const predictionMutation = usePrediction();
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setErrorMsg(null);
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      if (file.size > 10 * 1024 * 1024) {
        setErrorMsg('File size exceeds 10MB limit.');
        return;
      }
      setSelectedImage(file);
    }
  }, [setSelectedImage]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': [], 'image/png': [], 'image/webp': [] },
    maxFiles: 1,
  });

  const handlePredict = () => {
    if (selectedImage) {
      predictionMutation.mutate(selectedImage);
    }
  };

  const handleClear = (e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    setSelectedImage(null);
    setErrorMsg(null);
  };

  return (
    <div className="w-full space-y-6">
      
      {/* Dropzone Container */}
      <div 
        {...getRootProps()} 
        className={`relative border-2 border-dashed rounded-3xl p-8 sm:p-12 text-center cursor-pointer transition-all duration-200 ease-in-out shadow-xs
          ${isDragActive 
            ? 'border-blue-500 bg-blue-50/70 scale-[1.01]' 
            : 'border-slate-300 hover:border-blue-500/70 hover:bg-slate-50/80 bg-white'
          }
          ${selectedImage ? 'bg-slate-50/60 border-solid border-slate-200' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        {!selectedImage ? (
          <div className="flex flex-col items-center justify-center space-y-4">
            <div className="h-16 w-16 bg-blue-50 border border-blue-100 rounded-2xl flex items-center justify-center text-blue-600 shadow-xs">
              <UploadCloud className="h-8 w-8" />
            </div>
            <div className="space-y-1">
              <p className="text-lg font-bold text-slate-800">
                Drag & drop your dermatological photograph here
              </p>
              <p className="text-sm text-slate-500 font-medium">
                or click to browse local files from your computer
              </p>
            </div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-100 text-slate-500 text-xs font-semibold">
              <span>Supports JPEG, PNG, WEBP • Max: 10MB</span>
            </div>
            {errorMsg && (
              <p className="text-red-600 text-xs mt-2 font-semibold bg-red-50 px-3 py-1.5 rounded-lg border border-red-200">
                {errorMsg}
              </p>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center space-y-4">
            <div className="relative group max-w-sm rounded-2xl overflow-hidden border border-slate-200 shadow-md bg-white">
              <img 
                src={imagePreviewUrl!} 
                alt="Uploaded Lesion Preview" 
                className="max-h-72 w-auto object-contain mx-auto"
              />
              <button 
                onClick={handleClear}
                className="absolute top-3 right-3 p-2 bg-white/95 hover:bg-red-50 text-slate-600 hover:text-red-600 rounded-full shadow-md border border-slate-200 transition-colors"
                title="Remove image"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-slate-100 border border-slate-200 text-xs font-semibold text-slate-700">
              <FileImage className="h-4 w-4 text-blue-600" />
              <span>{selectedImage.name}</span>
              <span className="text-slate-400 font-normal">
                ({(selectedImage.size / 1024 / 1024).toFixed(2)} MB)
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Prominent Action Button Bar */}
      {selectedImage && (
        <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
          
          <button
            onClick={handlePredict}
            disabled={predictionMutation.isPending}
            className="inline-flex items-center justify-center gap-2.5 px-8 py-3.5 bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-700 hover:from-blue-700 hover:to-indigo-800 text-white font-bold text-sm rounded-2xl shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40 hover:-translate-y-0.5 transition-all disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none cursor-pointer"
          >
            {predictionMutation.isPending ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                <span>Running INT8 Dual Ensemble & Grad-CAM...</span>
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4" />
                <span>Run Clinical AI Diagnosis</span>
                <ArrowRight className="h-4 w-4" />
              </>
            )}
          </button>

          <button
            onClick={() => handleClear()}
            disabled={predictionMutation.isPending}
            className="inline-flex items-center gap-1.5 px-4 py-3.5 bg-white hover:bg-slate-50 text-slate-600 hover:text-slate-900 border border-slate-200 font-semibold text-sm rounded-2xl transition-all shadow-2xs disabled:opacity-50"
            title="Choose a different image"
          >
            <RotateCcw className="h-4 w-4 text-slate-400" />
            <span>Replace</span>
          </button>

        </div>
      )}
      
      {/* Error notification */}
      {predictionMutation.isError && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-2xl text-red-700 text-xs font-semibold text-center">
          An error occurred during diagnosis. Please check that the backend is running and try again.
        </div>
      )}

    </div>
  );
};

export default ImageDropzone;
