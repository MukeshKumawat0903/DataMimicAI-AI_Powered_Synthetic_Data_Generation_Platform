import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { ProfileResponse } from "@/lib/api/types";

// Backend response shape from POST /eda/profile
interface BackendProfileResponse {
  profile: Array<{
    column: string;
    dtype: string;
    "missing_%": number;
    n_unique: number;
    most_common?: string | number;
  }>;
  summary: ProfileResponse["summary"];
  suggestions: Array<{ column: string; action: string; fix_type: string }>;
}

function mapResponse(fileId: string, res: BackendProfileResponse): ProfileResponse {
  return {
    file_id: fileId,
    row_count: res.summary?.n_rows ?? 0,
    summary: res.summary,
    columns: (res.profile ?? []).map((p) => {
      const mine = (res.suggestions ?? []).filter((s) => s.column === p.column);
      return {
        column: p.column,
        dtype: p.dtype,
        missing_pct: p["missing_%"] ?? 0,
        unique_count: p.n_unique ?? 0,
        top_value: p.most_common,
        suggestions: mine.map((s) => s.action),
        fixes: mine.map((s) => ({ action: s.action, fix_type: s.fix_type })),
      };
    }),
  };
}

export function useProfile(fileId: string | null) {
  return useQuery<ProfileResponse>({
    queryKey: ["profile", fileId],
    queryFn: () =>
      apiClient
        .post<BackendProfileResponse>(`/eda/profile?file_id=${fileId}`, {})
        .then((r) => mapResponse(fileId as string, r.data)),
    enabled: !!fileId,
    staleTime: 5 * 60 * 1000,
  });
}
