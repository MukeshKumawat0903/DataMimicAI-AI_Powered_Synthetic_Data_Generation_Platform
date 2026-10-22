import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { CorrelationPair, CorrelationResponse } from "@/lib/api/types";

// Backend response shape from POST /eda/correlation
interface BackendCorrelationResponse {
  corr_heatmap_base64?: string;
  /** [col_a, col_b, corr] — each pair appears twice (a,b) and (b,a) */
  top_corrs?: Array<[string, string, number]>;
  /** [col_a, col_b, corr] for near-perfect correlations */
  leakage_pairs?: Array<[string, string, number]>;
}

const pairKey = (a: string, b: string) => [a, b].sort().join("\u0000");

function mapResponse(res: BackendCorrelationResponse): CorrelationResponse {
  const leakage = new Set((res.leakage_pairs ?? []).map(([a, b]) => pairKey(a, b)));
  const seen = new Set<string>();
  const top_pairs: CorrelationPair[] = [];
  for (const [col_a, col_b, correlation] of res.top_corrs ?? []) {
    const key = pairKey(col_a, col_b);
    if (col_a === col_b || seen.has(key)) continue;
    seen.add(key);
    top_pairs.push({ col_a, col_b, correlation, is_leakage: leakage.has(key) });
  }
  return { heatmap_base64: res.corr_heatmap_base64 ?? "", top_pairs };
}

export function useCorrelation(fileId: string | null) {
  return useQuery<CorrelationResponse>({
    queryKey: ["correlation", fileId],
    queryFn: () =>
      apiClient
        .post<BackendCorrelationResponse>(`/eda/correlation?file_id=${fileId}`, {})
        .then((r) => mapResponse(r.data)),
    enabled: !!fileId,
    staleTime: 5 * 60 * 1000,
  });
}
