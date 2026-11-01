import { apiClient } from './axios';
import type { PredictionResponse, BatchPredictRequest, BatchPredictionResponse } from '../types/prediction';

export const predictImage = async (file: File): Promise<PredictionResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post<PredictionResponse>('/api/v1/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

export const batchPredict = async (data: BatchPredictRequest): Promise<BatchPredictionResponse> => {
  const response = await apiClient.post<BatchPredictionResponse>('/api/v1/batch-predict', data);
  return response.data;
};
