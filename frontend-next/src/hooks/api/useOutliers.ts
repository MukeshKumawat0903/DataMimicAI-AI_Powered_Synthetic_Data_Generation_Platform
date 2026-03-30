import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { OutliersResponse, RemediateRequest } from "@/lib/api/types";

export function useOutliers(fileId: string | null, methods: string[] = ["iqr"]) {
  return useQuery<OutliersResponse>({
    queryKey: ["outliers", fileId, methods],
    queryFn: () =>
      apiClient
        // Backend uses Query params: file_id and methods as comma-separated string
        .post<OutliersResponse>(`/eda/outliers/detect-comprehensive`, null, {
          params: {
            file_id: fileId,
            methods: methods.join(","),
          },
        })
        .then((r) => r.data),
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
