import { useMutation, useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { PetMappingResponse, TransformDecision, TransformationComparison } from "@/lib/api/types";

interface ApplyResponse {
  status: string;
  summary: {
    applied_count: number;
    skipped_count: number;
    columns_modified: string[];
    skipped_actions: { column: string; transformation: string; reason: string }[];
  };
  /** The transformed dataset as gzip + base64 CSV (absent if too large) */
  transformed_data?: { content?: string; encoding?: string; compression?: string; reason?: string };
}

export interface SavedTransformConfig {
  filepath: string;
  filename: string;
  created_at: string;
  decision_count: number;
}

// Records which side of a utility/privacy conflict the user chose
export function useResolveConflict() {
  return useMutation({
    mutationFn: (input: { fileId: string; column: string; category: "utility" | "privacy"; transformation: string }) =>
      apiClient
        .post(`/eda/resolve-conflict`, null, {
          params: {
            file_id: input.fileId,
            column: input.column,
            chosen_category: input.category,
            chosen_transformation: input.transformation,
          },
        })
        .then((r) => r.data),
  });
}

// Before/after metrics for each accepted transformation (on a sample of rows)
export function useTransformComparisons() {
  return useMutation({
    mutationFn: (input: { fileId: string; decisions: TransformDecision[] }) =>
      apiClient
        .post<{ comparisons: TransformationComparison[] }>(
          `/eda/get-transform-comparisons`,
          { decisions: input.decisions, row_sampling_method: "random", row_sample_size: 1000, sampling_seed: 42 },
          { params: { file_id: input.fileId } }
        )
        .then((r) => r.data.comparisons ?? []),
  });
}

export function usePetMapping() {
  return useMutation({
    mutationFn: (input: { fileId: string; decisions: TransformDecision[] }) =>
      apiClient
        .post<PetMappingResponse>(`/eda/get-pet-mapping`, { decisions: input.decisions }, { params: { file_id: input.fileId } })
        .then((r) => r.data),
  });
}

// Applies the decisions to a copy of the data and records the config (the uploaded file isn't changed)
export function useApplyTransforms() {
  return useMutation({
    mutationFn: (input: { fileId: string; decisions: TransformDecision[] }) =>
      apiClient
        .post<ApplyResponse>(`/eda/apply-transform-config`, { decisions: input.decisions }, { params: { file_id: input.fileId } })
        .then((r) => r.data),
  });
}

export function useTransformCode() {
  return useMutation({
    mutationFn: (input: { fileId: string; decisions: TransformDecision[] }) =>
      apiClient
        .post<{ code: string }>(`/eda/get-transform-batch-code`, { decisions: input.decisions }, { params: { file_id: input.fileId } })
        .then((r) => r.data.code),
  });
}

export function useSaveTransformConfig() {
  return useMutation({
    mutationFn: (input: { fileId: string; decisions: TransformDecision[] }) =>
      apiClient
        .post<{ config_path: string }>(`/eda/save-transform-config`, { decisions: input.decisions }, { params: { file_id: input.fileId } })
        .then((r) => r.data),
  });
}

export function useTransformConfigs(fileId: string | null, enabled: boolean) {
  return useQuery<SavedTransformConfig[]>({
    queryKey: ["transform-configs", fileId],
    queryFn: () =>
      apiClient
        .get<{ configs: SavedTransformConfig[] }>(`/eda/list-transform-configs`, { params: { file_id: fileId } })
        .then((r) => r.data.configs ?? []),
    enabled: !!fileId && enabled,
  });
}

export function useLoadTransformConfig() {
  return useMutation({
    mutationFn: (input: { fileId: string; path: string }) =>
      apiClient
        .get<{ config: { transformations: TransformDecision[] } }>(`/eda/load-transform-config`, {
          params: { file_id: input.fileId, config_path: input.path },
        })
        .then((r) => r.data.config.transformations ?? []),
  });
}

/** Turn the gzip + base64 CSV payload from apply-transform-config into a downloadable Blob */
export async function decodeTransformedData(content: string): Promise<Blob> {
  const bytes = Uint8Array.from(atob(content), (c) => c.charCodeAt(0));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Blob([await new Response(stream).arrayBuffer()], { type: "text/csv" });
}
