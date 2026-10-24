import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import { useWorkspaceStore } from "@/lib/store/workspace";
import type {
  AcfPacfResponse,
  CooksDistanceResponse,
  DecompositionResponse,
  DependenciesResponse,
  DistributionsResponse,
  KAnonymityResponse,
  MutualInfoResponse,
  QuasiIdentifier,
} from "@/lib/api/types";

export type CleaningKind = "impute" | "encode" | "drop_outliers" | "drop_column";

const CLEANING_ENDPOINT: Record<CleaningKind, string> = {
  impute: "/eda/fix-missing",
  encode: "/eda/encode-high-cardinality",
  drop_outliers: "/eda/drop-outliers",
  drop_column: "/eda/drop-columns",
};

// In-place data cleaning; every cached result for the dataset becomes stale afterwards
export function useCleaningAction() {
  const qc = useQueryClient();
  const setLastChangedColumns = useWorkspaceStore((s) => s.setLastChangedColumns);
  return useMutation({
    mutationFn: (input: { fileId: string; kind: CleaningKind; column: string; method?: string }) =>
      apiClient
        .post(CLEANING_ENDPOINT[input.kind], null, {
          params: {
            file_id: input.fileId,
            columns: input.column,
            ...(input.kind === "impute" ? { method: input.method ?? "median" } : {}),
            ...(input.kind === "encode" ? { max_values: 10 } : {}),
            ...(input.kind === "drop_outliers" ? { z_thresh: 3 } : {}),
          },
        })
        .then((r) => r.data),
    onSuccess: (_, input) => {
      setLastChangedColumns(input.kind === "drop_column" ? [] : [input.column]);
      qc.invalidateQueries({ predicate: (q) => q.queryKey.includes(input.fileId) });
    },
  });
}

// Restore the original upload (undoes cleaning, remediation and transform configs)
export function useResetDataset() {
  const qc = useQueryClient();
  const { setDecisions, setFeedback, setLastChangedColumns, setKpiSummary } = useWorkspaceStore();
  return useMutation({
    mutationFn: (fileId: string) =>
      apiClient
        .post<{ data_restored: boolean; message: string }>(`/eda/rollback-transforms`, { confirmed: true }, { params: { file_id: fileId } })
        .then((r) => r.data),
    onSuccess: (_, fileId) => {
      setDecisions([]);
      setFeedback([]);
      setLastChangedColumns([]);
      setKpiSummary(null);
      qc.invalidateQueries({ predicate: (q) => q.queryKey.includes(fileId) });
    },
  });
}

export function useDistributions() {
  return useMutation({
    mutationFn: (fileId: string) =>
      apiClient.post<DistributionsResponse>(`/eda/compute-distributions/${fileId}`).then((r) => r.data),
  });
}

export function useMutualInfo() {
  return useMutation({
    mutationFn: (fileId: string) =>
      apiClient
        .post<MutualInfoResponse>(`/eda/compute-mi-matrix/${fileId}`, { columns: null, normalize: true })
        .then((r) => r.data),
  });
}

export function useDependencies() {
  return useMutation({
    mutationFn: (fileId: string) =>
      apiClient.get<DependenciesResponse>(`/eda/detect-dependencies/${fileId}`).then((r) => r.data),
  });
}

export function useCooksDistance() {
  return useMutation({
    mutationFn: (input: { fileId: string; target: string; features: string[] }) =>
      apiClient
        .post<CooksDistanceResponse>(`/eda/outliers/detect-cooks-distance`, null, {
          params: { file_id: input.fileId, target_col: input.target, feature_cols: input.features.join(",") },
        })
        .then((r) => r.data.stats?.[0]),
  });
}

export function useQuasiIdentifiers() {
  return useMutation({
    mutationFn: (fileId: string) =>
      apiClient
        .get<{ potential_qis: QuasiIdentifier[] }>(`/eda/suggest-quasi-identifiers/${fileId}`)
        .then((r) => r.data.potential_qis ?? []),
  });
}

export function useKAnonymity() {
  return useMutation({
    mutationFn: (input: { fileId: string; quasiIdentifiers: string[]; k: number }) =>
      apiClient
        .post<KAnonymityResponse>(`/eda/compute-k-anonymity`, {
          file_id: input.fileId,
          quasi_identifiers: input.quasiIdentifiers,
          k_threshold: input.k,
        })
        .then((r) => r.data),
  });
}

export function useAcfPacf() {
  return useMutation({
    mutationFn: (input: { fileId: string; datetimeCol: string; targetCol: string }) =>
      apiClient
        .post<AcfPacfResponse>(`/eda/timeseries-acf-pacf`, {
          file_id: input.fileId,
          datetime_col: input.datetimeCol,
          target_col: input.targetCol,
          lags: 30,
        })
        .then((r) => r.data.acf_pacf),
  });
}

export function useDecomposition() {
  return useMutation({
    mutationFn: (input: { fileId: string; datetimeCol: string; targetCol: string }) =>
      apiClient
        .post<DecompositionResponse>(`/eda/timeseries-decompose`, {
          file_id: input.fileId,
          datetime_col: input.datetimeCol,
          target_col: input.targetCol,
          model: "additive",
        })
        .then((r) => r.data.decomposition),
  });
}

export interface TimeseriesOutliers {
  outlier_indices: Record<string, string[]>;
  stats: { outlier_count: number; outlier_percent: number; threshold: number }[];
}

// Points that stand out from the trend + seasonality (STL residuals)
export function useTimeseriesOutliers() {
  return useMutation({
    mutationFn: (input: { fileId: string; datetimeCol: string; targetCol: string }) =>
      apiClient
        .post<TimeseriesOutliers>(`/eda/outliers/detect-timeseries`, null, {
          params: { file_id: input.fileId, value_col: input.targetCol, datetime_col: input.datetimeCol },
        })
        .then((r) => r.data),
  });
}
