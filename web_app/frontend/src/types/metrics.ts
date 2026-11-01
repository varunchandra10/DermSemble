export interface HealthResponse {
  status: string;
  message: string;
}

export interface ModelInfoResponse {
  models_loaded: string[];
  versions: Record<string, string>;
  paths: Record<string, string>;
}
