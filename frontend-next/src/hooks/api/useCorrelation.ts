import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { CorrelationMethod, CorrelationPair, CorrelationResponse } from "@/lib/api/types";

type Matrix = Record<string, Record<string, number>>;

// Backend response shape from POST /eda/correlation
interface BackendCorrelationResponse {
  corr_heatmap_base64?: string;
  pearson_corr_matrix?: Matrix;
  spearman_corr_matrix?: Matrix;
  kendall_corr_matrix?: Matrix;
  /** [col_a, col_b, corr] — each pair appears twice (a,b) and (b,a) */
  top_corrs?: Array<[string, string, number]>;
  /** [col_a, col_b, corr] for near-perfect correlations */
  leakage_pairs?: Array<[string, string, number]>;
  categorical_assoc?: Array<{ "Feature 1": string; "Feature 2": string; CramersV: number }>;
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
  return {
    heatmap_base64: res.corr_heatmap_base64 ?? "",
    top_pairs,
    matrices: {
      pearson: res.pearson_corr_matrix ?? {},
      spearman: res.spearman_corr_matrix ?? {},
      kendall: res.kendall_corr_matrix ?? {},
    } satisfies Record<CorrelationMethod, Matrix>,
    categorical: (res.categorical_assoc ?? [])
      .map((c) => ({ col_a: c["Feature 1"], col_b: c["Feature 2"], cramers_v: c.CramersV }))
      .sort((a, b) => b.cramers_v - a.cramers_v),
  };
}

export function useCorrelation(fileId: string | null, topK = 10) {
  return useQuery<CorrelationResponse>({
    queryKey: ["correlation", fileId, topK],
    queryFn: () =>
      apiClient
        // top_corrs lists each pair twice, so ask for 2x to get topK distinct pairs
        .post<BackendCorrelationResponse>(`/eda/correlation`, {}, { params: { file_id: fileId, top_k: topK * 2 } })
        .then((r) => mapResponse(r.data)),
    enabled: !!fileId,
    staleTime: 5 * 60 * 1000,
  });
}
