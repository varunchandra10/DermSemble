import { create } from 'zustand';
import type { PredictionResponse } from '../types/prediction';
import type { HealthResponse } from '../types/metrics';

interface AppState {
  // Image State
  selectedImage: File | null;
  imagePreviewUrl: string | null;
  setSelectedImage: (file: File | null) => void;

  // Prediction State
  predictionResult: PredictionResponse | null;
  setPredictionResult: (result: PredictionResponse | null) => void;
  
  // Health State
  backendHealth: HealthResponse | null;
  setBackendHealth: (health: HealthResponse | null) => void;

  // Reset
  resetDiagnosis: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  selectedImage: null,
  imagePreviewUrl: null,
  setSelectedImage: (file) => {
    if (file) {
      const url = URL.createObjectURL(file);
      set({ selectedImage: file, imagePreviewUrl: url });
    } else {
      set((state) => {
        if (state.imagePreviewUrl) {
          URL.revokeObjectURL(state.imagePreviewUrl);
        }
        return { selectedImage: null, imagePreviewUrl: null };
      });
    }
  },

  predictionResult: null,
  setPredictionResult: (result) => set({ predictionResult: result }),

  backendHealth: null,
  setBackendHealth: (health) => set({ backendHealth: health }),

  resetDiagnosis: () => set((state) => {
    if (state.imagePreviewUrl) {
      URL.revokeObjectURL(state.imagePreviewUrl);
    }
    return {
      selectedImage: null,
      imagePreviewUrl: null,
      predictionResult: null,
    };
  }),
}));
