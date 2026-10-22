import { useMutation } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import { useWorkspaceStore } from "@/lib/store/workspace";
import type { GenerationRequest } from "@/lib/api/types";

export function useGenerate() {
  const setGeneratedFileId = useWorkspaceStore((s) => s.setGeneratedFileId);

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
      // Trigger file download in browser
      const url = URL.createObjectURL(response.data as Blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `synthetic_data_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      return { synthetic_file_id: synFileId };
    },
    onSuccess: ({ synthetic_file_id }) => {
      if (synthetic_file_id) {
        setGeneratedFileId(synthetic_file_id);
      }
    },
  });
}
