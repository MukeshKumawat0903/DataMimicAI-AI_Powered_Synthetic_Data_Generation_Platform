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
  return {
    file_id: fileId,
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
        result = await apiClient
          .post(`/eda/outliers/remediate`, {}, {
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
      qc.invalidateQueries({ queryKey: ["outliers", vars.file_id] });
      qc.invalidateQueries({ queryKey: ["dataset", vars.file_id] });
    },
  });
}
