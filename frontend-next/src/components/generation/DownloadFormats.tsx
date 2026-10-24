"use client";

import { apiClient } from "@/lib/api/client";
import { downloadBlob } from "@/lib/download";
import { Button } from "@/components/ui/button";
import { Download } from "lucide-react";
import { toast } from "sonner";
import type { AppError } from "@/lib/api/types";

const FORMATS = [
  { format: "csv", label: "CSV" },
  { format: "json", label: "JSON" },
  { format: "parquet", label: "Parquet" },
] as const;

/** Download the current synthetic dataset in the chosen format */
export function DownloadFormats({ fileId }: { fileId: string }) {
  async function download(format: (typeof FORMATS)[number]["format"]) {
    try {
      const res = await apiClient.get(`/eda/download`, {
        params: { file_id: `syn_${fileId}`, format },
        responseType: "blob",
      });
      downloadBlob(res.data as Blob, `synthetic_data.${format}`);
      toast.success("Download started", { description: `synthetic_data.${format}` });
    } catch (e) {
      toast.error("Download failed", { description: (e as Partial<AppError>).message });
    }
  }

  return (
    <div className="flex items-center gap-1">
      <Download className="h-3.5 w-3.5 text-muted-foreground" aria-hidden />
      {FORMATS.map((f) => (
        <Button key={f.format} variant="outline" size="sm" className="h-7 px-2 text-xs" onClick={() => download(f.format)}>
          {f.label}
        </Button>
      ))}
    </div>
  );
}
