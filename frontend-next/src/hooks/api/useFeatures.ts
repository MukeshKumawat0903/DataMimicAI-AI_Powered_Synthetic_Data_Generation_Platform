import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { FeaturesResponse } from "@/lib/api/types";

export function useFeatures(fileId: string | null) {
  return useQuery<FeaturesResponse>({
    queryKey: ["features", fileId],
    queryFn: () =>
      apiClient
        // context-aware-suggestions returns utility_suggestions, privacy_suggestions, and conflicts
        .post<FeaturesResponse>(`/eda/context-aware-suggestions`, null, {
          params: { file_id: fileId },
        })
        .then((r) => r.data),
    enabled: !!fileId,
    staleTime: 5 * 60 * 1000,
  });
}
