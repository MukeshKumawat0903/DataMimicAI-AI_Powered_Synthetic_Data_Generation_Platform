"use client";

import { useCallback, useRef, useState } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Upload, FileText, X, AlertCircle } from "lucide-react";
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
              "flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-10 text-center cursor-pointer transition-colors",
              isDragActive
                ? "border-primary bg-primary/5"
                : "border-muted-foreground/30 hover:border-primary/60 hover:bg-accent/30"
            )}
          >
            <input {...getInputProps()} />
            <Upload
              className={cn(
                "h-10 w-10 mb-3",
                isDragActive ? "text-primary" : "text-muted-foreground"
              )}
            />
            <p className="text-sm font-medium">
              {isDragActive
                ? "Drop your CSV here…"
                : "Drag & drop a CSV file, or click to browse"}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Maximum size: {formatBytes(MAX_SIZE)}
            </p>
          </div>
        )}

        {/* Selected file preview */}
        {selectedFile && (
          <div className="flex items-center justify-between rounded-lg border px-4 py-3 bg-accent/40">
            <div className="flex items-center gap-3">
              <FileText className="h-5 w-5 text-primary shrink-0" />
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
          <Alert variant="default" className="border-green-500/50 bg-green-50 dark:bg-green-950/20">
            {/* P3.3: tabIndex -1 so focus() can target it programmatically */}
            <AlertDescription
              ref={successRef}
              tabIndex={-1}
              className="text-green-700 dark:text-green-400 text-sm outline-none"
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
