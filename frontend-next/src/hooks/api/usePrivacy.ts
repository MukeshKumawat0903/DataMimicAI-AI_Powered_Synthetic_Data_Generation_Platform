import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { PrivacyResponse } from "@/lib/api/types";

export function usePrivacy(fileId: string | null, mode: "fast" | "deep" = "fast") {
  return useQuery<PrivacyResponse>({
    queryKey: ["privacy", fileId, mode],
    queryFn: () => {
      if (mode === "deep") {
        return apiClient
          .post<PrivacyResponse>(`/eda/pii-scan-deep`, null, {
            params: { file_id: fileId },
          })
          .then((r) => r.data);
      }
      return apiClient
        .post<PrivacyResponse>(`/eda/pii-scan-fast/${fileId}`, null)
        .then((r) => r.data);
    },
    enabled: !!fileId,
    staleTime: 10 * 60 * 1000,
  });
}
