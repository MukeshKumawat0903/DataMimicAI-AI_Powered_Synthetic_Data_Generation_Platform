import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import { useWorkspaceStore } from "@/lib/store/workspace";
import type { GenerationRequest } from "@/lib/api/types";

export function useGenerate() {
  const qc = useQueryClient();
  const setGeneratedFileId = useWorkspaceStore((s) => s.setGeneratedFileId);
  const addGeneration = useWorkspaceStore((s) => s.addGeneration);

  return useMutation({
    mutationFn: async (req: GenerationRequest) => {
      const response = await apiClient.post(
        `/feedback-generation/generate_and_download`,
        {
          feedback: req.feedback,
          generator_config: req.generator_config,
        },
        {
          params: {
            file_id: req.file_id,
          },
          responseType: "blob",
          timeout: 600_000, // 10 min — model training can be slow
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      // Backend saves the synthetic data as syn_<file_id>.csv, so the source
      // file_id is what /eda/download and /metrics need later.
      const synFileId = req.file_id;
      const versionId = response.headers["x-synthetic-version"] as string | undefined;
      // Trigger file download in browser
      const url = URL.createObjectURL(response.data as Blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `synthetic_data_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      return { synthetic_file_id: synFileId, versionId };
    },
    onSuccess: ({ synthetic_file_id, versionId }, req) => {
      if (synthetic_file_id) {
        setGeneratedFileId(synthetic_file_id);
      }
      addGeneration({
        algorithm: req.generator_config.algorithm,
        numRows: req.generator_config.num_rows,
        epochs: req.generator_config.epochs,
        timestamp: new Date().toISOString(),
        versionId,
      });
      // The synthetic file was replaced, so cached scores and charts are stale
      qc.invalidateQueries({
        predicate: (q) => ["quality", "viz"].includes(q.queryKey[0] as string) && q.queryKey[1] === req.file_id,
      });
    },
  });
}

// Make an earlier generation the current synthetic dataset again
export function useRestoreVersion() {
  const qc = useQueryClient();
  const setCurrentVersionId = useWorkspaceStore((s) => s.setCurrentVersionId);
  return useMutation({
    mutationFn: (input: { fileId: string; versionId: string }) =>
      apiClient
        .post(`/feedback-generation/restore-version`, null, {
          params: { file_id: input.fileId, version: input.versionId },
        })
        .then((r) => r.data),
    onSuccess: (_, input) => {
      setCurrentVersionId(input.versionId);
      qc.invalidateQueries({
        predicate: (q) => ["quality", "viz"].includes(q.queryKey[0] as string) && q.queryKey[1] === input.fileId,
      });
    },
  });
}

export interface ModelEvaluation {
  summary: { total_models: number; successful_models: number; failed_models: number; total_evaluation_time: number };
  metrics: Array<{
    Model: string;
    Status: string;
    Error?: string | null;
    KolmogorovSmirnov: number | null;
    JensenShannon: number | null;
    Wasserstein: number | null;
    MaxMeanDiscrep: number | null;
    Training_Time_Seconds?: string | null;
  }>;
  best_model: string;
}

// AutoML: train every SynthCity model briefly and rank them (slow; needs free memory)
export function useEvaluateModels() {
  return useMutation({
    mutationFn: (input: { fileId: string; epochs: number }) =>
      apiClient
        .post<ModelEvaluation>(`/evaluate_models`, null, {
          params: { file_id: input.fileId, epochs: input.epochs },
          timeout: 1_800_000,
        })
        .then((r) => r.data),
  });
}
