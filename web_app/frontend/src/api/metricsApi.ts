import { apiClient } from './axios';
import type { HealthResponse, ModelInfoResponse } from '../types/metrics';

export const getHealth = async (): Promise<HealthResponse> => {
  const response = await apiClient.get<HealthResponse>('/api/v1/health');
  return response.data;
};

export const getModelInfo = async (): Promise<ModelInfoResponse> => {
  const response = await apiClient.get<ModelInfoResponse>('/api/v1/model-info');
  return response.data;
};

export const getMetrics = async (): Promise<any> => {
  const response = await apiClient.get('/api/v1/metrics');
  return response.data;
};
