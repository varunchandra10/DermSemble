import { useQuery } from '@tanstack/react-query';
import { getHealth, getModelInfo } from '../api/metricsApi';
import { useAppStore } from '../app/store';

export const useHealthCheck = () => {
  const setBackendHealth = useAppStore((state) => state.setBackendHealth);

  return useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      const data = await getHealth();
      setBackendHealth(data);
      return data;
    },
    refetchInterval: 60000, // Check every minute
    retry: 2,
  });
};

export const useModelInfo = () => {
  return useQuery({
    queryKey: ['modelInfo'],
    queryFn: getModelInfo,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};
