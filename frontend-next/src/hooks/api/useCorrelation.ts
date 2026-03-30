import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { CorrelationResponse } from "@/lib/api/types";

export function useCorrelation(fileId: string | null) {
  return useQuery<CorrelationResponse>({
    queryKey: ["correlation", fileId],
    queryFn: () =>
      apiClient
        .post<CorrelationResponse>(`/eda/correlation?file_id=${fileId}`, {})
        .then((r) => r.data),
    enabled: !!fileId,
    staleTime: 5 * 60 * 1000,
  });
}
