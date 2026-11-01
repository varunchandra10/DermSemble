import { useMutation } from '@tanstack/react-query';
import { predictImage } from '../api/predictionApi';
import { useAppStore } from '../app/store';

export const usePrediction = () => {
  const setPredictionResult = useAppStore((state) => state.setPredictionResult);

  return useMutation({
    mutationFn: (file: File) => predictImage(file),
    onSuccess: (data) => {
      setPredictionResult(data);
    },
    onError: (error) => {
      console.error('Prediction failed:', error);
      // Can show toast here later
    },
  });
};
