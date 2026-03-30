import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { ProfileResponse } from "@/lib/api/types";

export function useProfile(fileId: string | null) {
  return useQuery<ProfileResponse>({
    queryKey: ["profile", fileId],
    queryFn: () =>
      apiClient
        .post<ProfileResponse>(`/eda/profile?file_id=${fileId}`, {})
        .then((r) => r.data),
    enabled: !!fileId,
    staleTime: 5 * 60 * 1000,
  });
}
