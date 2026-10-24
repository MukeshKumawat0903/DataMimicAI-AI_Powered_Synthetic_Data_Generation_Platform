"use client";

import { useCallback, useRef, useState } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { UploadCloud, FileSpreadsheet, X, AlertCircle, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useUpload } from "@/hooks/api/useUpload";
import { toast } from "sonner";
import type { AppError } from "@/lib/api/types";

const MAX_SIZE =
  Number(process.env.NEXT_PUBLIC_MAX_UPLOAD_SIZE) || 10 * 1024 * 1024;

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function FileUploader() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  // P3.3: ref for focus management after upload success
  const successRef = useRef<HTMLDivElement>(null);

  const { mutate: upload, isPending, error, isSuccess } = useUpload();

  const onDrop = useCallback((accepted: File[], rejected: FileRejection[]) => {
    setValidationError(null);

    if (rejected.length > 0) {
      setValidationError("Only CSV files are supported.");
      return;
    }

    const file = accepted[0];
    if (!file) return;

    if (file.size > MAX_SIZE) {
      setValidationError(
        `File too large (${formatBytes(file.size)}). Maximum allowed is ${formatBytes(MAX_SIZE)}.`
      );
      return;
    }

    setSelectedFile(file);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    multiple: false,
    maxSize: MAX_SIZE,
    noClick: !!selectedFile,
  });

  function handleUpload() {
    if (!selectedFile) return;
    upload(selectedFile, {
      // P0.2: Toast on upload success
      onSuccess: (data) => {
        toast.success("Dataset uploaded", {
          description: `${data.num_rows ?? ""} rows · ${data.columns?.length ?? ""} columns detected.`,
        });
        // P3.3: Return focus to success message
        setTimeout(() => successRef.current?.focus(), 50);
      },
      onError: (err) => {
        const appErr = err as AppError;
        toast.error("Upload failed", { description: appErr?.message });
      },
    });
  }

  function handleRemove() {
    setSelectedFile(null);
    setValidationError(null);
  }

  const uploadError = error as AppError | null;
  const displayError = validationError ?? uploadError?.message ?? null;

  return (
    <Card className="w-full">
      <CardContent className="pt-6 space-y-4">
        {/* Drop zone */}
        {!selectedFile && (
          <div
            {...getRootProps()}
            className={cn(
              "group flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-12 text-center cursor-pointer transition-all duration-200",
              isDragActive
                ? "border-primary bg-primary/10 scale-[1.01]"
                : "border-primary/25 bg-gradient-to-b from-primary/[0.04] to-transparent hover:border-primary/60 hover:bg-primary/[0.06]"
            )}
          >
            <input {...getInputProps()} />
            <span
              className={cn(
                "bg-brand-gradient mb-4 flex h-14 w-14 items-center justify-center rounded-2xl text-white shadow-lg shadow-indigo-500/30 transition-transform duration-200",
                isDragActive ? "scale-110" : "group-hover:-translate-y-0.5"
              )}
            >
              <UploadCloud className="h-7 w-7" />
            </span>
            <p className="text-base font-semibold">
              {isDragActive
                ? "Drop your CSV here…"
                : "Drag & drop a CSV file, or click to browse"}
            </p>
            <div className="mt-3 flex flex-wrap justify-center gap-2 text-xs text-muted-foreground">
              <span className="rounded-full bg-muted px-2.5 py-0.5">CSV only</span>
              <span className="rounded-full bg-muted px-2.5 py-0.5">Maximum size: {formatBytes(MAX_SIZE)}</span>
              <span className="rounded-full bg-muted px-2.5 py-0.5">Header row required</span>
            </div>
          </div>
        )}

        {/* Selected file preview */}
        {selectedFile && (
          <div className="flex items-center justify-between rounded-xl border bg-accent/50 px-4 py-3">
            <div className="flex items-center gap-3">
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-100 text-emerald-600 dark:bg-emerald-500/15 dark:text-emerald-300">
                <FileSpreadsheet className="h-5 w-5" />
              </span>
              <div className="min-w-0">
                <p className="text-sm font-medium truncate">{selectedFile.name}</p>
                <p className="text-xs text-muted-foreground">
                  {formatBytes(selectedFile.size)}
                </p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleRemove}
              disabled={isPending}
              aria-label="Remove selected file"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        )}

        {/* Upload progress */}
        {isPending && (
          <div className="space-y-1">
            <Progress value={null} className="h-2 animate-pulse" />
            <p className="text-xs text-muted-foreground text-center">
              Uploading…
            </p>
          </div>
        )}

        {/* Success message */}
        {isSuccess && (
          <Alert variant="default" className="border-emerald-500/40 bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300">
            <CheckCircle2 className="h-4 w-4" />
            {/* P3.3: tabIndex -1 so focus() can target it programmatically */}
            <AlertDescription
              ref={successRef}
              tabIndex={-1}
              className="text-emerald-700 dark:text-emerald-300 text-sm font-medium outline-none"
            >
              Dataset uploaded successfully!
            </AlertDescription>
          </Alert>
        )}

        {/* Error message */}
        {displayError && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{displayError}</AlertDescription>
          </Alert>
        )}

        {/* Upload action */}
        {selectedFile && !isSuccess && (
          <Button
            size="lg"
            className="w-full"
            onClick={handleUpload}
            disabled={isPending}
          >
            {isPending ? "Uploading…" : "Upload Dataset"}
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
