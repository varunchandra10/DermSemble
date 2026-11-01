export interface ClassProbability {
  class_name: string;
  probability: number;
}

export interface PredictionResponse {
  prediction: string;
  confidence: number;
  top_3_predictions: ClassProbability[];
  class_probabilities: ClassProbability[];
  report: string;
  heatmap_url: string | null;
  latency_ms: number;
  disclaimer: string;
}

export interface BatchPredictRequest {
  images: string[]; // Base64 encoded strings
}

export interface BatchPredictionResponse {
  predictions: PredictionResponse[];
}
