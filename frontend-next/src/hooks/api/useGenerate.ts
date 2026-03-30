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
      // Build synthetic file ID from the Content-Disposition header or derive it
      const contentDisposition = response.headers["content-disposition"] as string | undefined;
      let synFileId: string | null = null;
      if (contentDisposition) {
        const match = contentDisposition.match(/filename[^;=\n]*=([^;\n]*)/);
        if (match) {
          // e.g. syn_<uuid>.csv → strip syn_ and .csv
          const fname = match[1].replace(/['"]/g, "").trim();
          synFileId = fname.replace(/^syn_/, "").replace(/\.csv$/, "");
        }
      }
      if (!synFileId) {
        synFileId = `local_${Date.now()}`;
      }
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
