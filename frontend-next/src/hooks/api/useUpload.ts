import { useMutation } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { UploadResponse, LoadDemoResponse, AppError } from "@/lib/api/types";
import { useWorkspaceStore } from "@/lib/store/workspace";

// ─── POST /upload ─────────────────────────────────────────────────────────────

export function useUpload() {
  const setUploadResult = useWorkspaceStore((s) => s.setUploadResult);

  return useMutation<UploadResponse, AppError, File>({
    mutationFn: async (file: File) => {
      const form = new FormData();
      form.append("file", file);
      const res = await apiClient.post<UploadResponse>("/upload", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      return res.data;
    },
    retry: 1,
    onSuccess: (data) => {
      setUploadResult({
        fileId: data.file_id,
        columns: data.columns,
        rowCount: data.num_rows,
      });
    },
  });
}

// ─── POST /load-demo ──────────────────────────────────────────────────────────

export function useLoadDemo() {
  const setUploadResult = useWorkspaceStore((s) => s.setUploadResult);

  return useMutation<LoadDemoResponse, AppError, string>({
    mutationFn: async (algorithm: string) => {
      const res = await apiClient.post<LoadDemoResponse>("/load-demo", null, {
        params: { algorithm },
      });
      return res.data;
    },
    onSuccess: (data) => {
      setUploadResult({
        fileId: data.file_id,
        columns: data.columns,
        rowCount: data.num_rows,
      });
    },
  });
}
