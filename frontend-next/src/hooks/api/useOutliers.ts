import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { OutliersResponse, RemediateRequest } from "@/lib/api/types";

// Backend response shape from POST /eda/outliers/detect-comprehensive
interface BackendOutliersResponse {
  /** method → column → outlier row indices */
  results: Record<string, Record<string, number[]>>;
  stats: Array<{
    column: string;
    method: string;
    outlier_count: number;
    outlier_percent: number;
  }>;
}

function mapResponse(fileId: string, res: BackendOutliersResponse): OutliersResponse {
  const indices: Record<string, number[]> = {};
  for (const byColumn of Object.values(res.results ?? {})) {
    for (const [column, rows] of Object.entries(byColumn ?? {})) {
      indices[column] = Array.from(new Set([...(indices[column] ?? []), ...rows]));
    }
  }
  return {
    file_id: fileId,
    indices,
    methods_used: Object.keys(res.results ?? {}),
    // Only columns that actually have outliers are worth showing / remediating
    results: (res.stats ?? [])
      .filter((s) => s.outlier_count > 0)
      .map((s) => ({
        column: s.column,
        method: s.method,
        outlier_count: s.outlier_count,
        pct_affected: s.outlier_percent ?? 0,
      })),
  };
}

// UI method labels → method names the backend accepts
const BACKEND_METHOD: Record<string, string> = {
  IQR: "iqr",
  ZScore: "zscore",
  MAD: "mad",
  IsolationForest: "isolation_forest",
  LOF: "lof",
};

export function useOutliers(fileId: string | null, methods: string[] = ["IQR"]) {
  return useQuery<OutliersResponse>({
    queryKey: ["outliers", fileId, methods],
    queryFn: () =>
      apiClient
        // Backend uses Query params: file_id and methods as comma-separated string
        .post<BackendOutliersResponse>(`/eda/outliers/detect-comprehensive`, null, {
          params: {
            file_id: fileId,
            methods: methods.map((m) => BACKEND_METHOD[m] ?? m.toLowerCase()).join(","),
          },
        })
        .then((r) => mapResponse(fileId as string, r.data)),
    enabled: !!fileId,
    staleTime: 5 * 60 * 1000,
  });
}

export function useRemediateOutliers() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (req: RemediateRequest) => {
      // Backend takes one method per request — group columns by action
      const grouped: Record<string, string[]> = {};
      for (const { column, action } of req.remediations) {
        (grouped[action] ??= []).push(column);
      }
      let result: unknown;
      for (const [method, columns] of Object.entries(grouped)) {
        const body: Record<string, unknown> = {
          // Save over the working dataset (like the other cleaning actions) so later steps use it
          output_file_id: req.file_id,
        };
        if (method === "remove") {
          body.outlier_indices = Object.fromEntries(columns.map((c) => [c, req.indices?.[c] ?? []]));
        }
        result = await apiClient
          .post(`/eda/outliers/remediate`, body, {
            params: {
              file_id: req.file_id,
              method,
              columns: columns.join(","),
            },
          })
          .then((r) => r.data);
      }
      return result;
    },
    onSuccess: (_, vars) => {
      // The dataset changed in place, so every cached result for it is stale
      qc.invalidateQueries({ predicate: (q) => q.queryKey.includes(vars.file_id) });
    },
  });
}
