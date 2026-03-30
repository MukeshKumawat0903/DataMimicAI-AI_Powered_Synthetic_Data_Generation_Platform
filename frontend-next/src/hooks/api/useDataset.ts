import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { DatasetResponse, AppError } from "@/lib/api/types";

export function useDataset(fileId: string | null) {
  return useQuery<DatasetResponse, AppError>({
    queryKey: ["dataset", fileId],
    queryFn: async () => {
      const res = await apiClient.get<DatasetResponse>(
        `/eda/get-data/${fileId}`
      );
      return res.data;
    },
    enabled: !!fileId,
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: 1,
  });
}
