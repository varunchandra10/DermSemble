import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptor for logging or handling global errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // We can show toast notifications here in a more advanced setup
    console.error('API Error:', error.response?.data?.detail || error.message);
    return Promise.reject(error);
  }
);
