"use client";

import { useRouter } from "next/navigation";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ArrowRight } from "lucide-react";
import { FileUploader } from "@/components/data/FileUploader";
import { DemoModeSelector } from "@/components/data/DemoModeSelector";
import { DataTable } from "@/components/data/DataTable";
import { KPIBanner } from "@/components/data/KPIBanner";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { useDataset } from "@/hooks/api/useDataset";

export default function UploadPage() {
  const router = useRouter();
  const fileId = useWorkspaceStore((s) => s.fileId);
  const kpiSummary = useWorkspaceStore((s) => s.kpiSummary);

  const { data: dataset, isLoading } = useDataset(fileId);

  return (
    <div className="mx-auto max-w-5xl px-6 py-8 space-y-8">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Upload Dataset</h1>
        <p className="text-muted-foreground mt-1 text-sm">
          Upload a CSV file or load a demo dataset to get started.
        </p>
      </div>

      {/* Upload / Demo section */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        {/* CSV Upload — wider column */}
        <div className="lg:col-span-3">
          <h2 className="text-sm font-semibold mb-3 text-muted-foreground uppercase tracking-wide">
            Upload your file
          </h2>
          <FileUploader />
        </div>

        {/* Divider */}
        <div className="hidden lg:flex flex-col items-center justify-center gap-2">
          <Separator orientation="vertical" className="flex-1" />
          <span className="text-xs font-medium text-muted-foreground bg-background px-2">
            OR
          </span>
          <Separator orientation="vertical" className="flex-1" />
        </div>

        <div className="lg:col-span-1 flex flex-col gap-4">
          {/* Demo mode */}
          <h2 className="text-sm font-semibold mb-3 text-muted-foreground uppercase tracking-wide">
            Use demo data
          </h2>
          <DemoModeSelector />
        </div>
      </div>

      {/* Data preview section — shown once a dataset is loaded */}
      {fileId && (
        <div className="space-y-5">
          <Separator />

          <div>
            <h2 className="text-lg font-semibold">Data Preview</h2>
            {dataset && (
              <p className="text-sm text-muted-foreground">
                {dataset.num_rows.toLocaleString()} rows ×{" "}
                {dataset.num_columns} columns
              </p>
            )}
          </div>

          {/* KPI Banner (only shown when KPI summary is available) */}
          {kpiSummary && <KPIBanner summary={kpiSummary} />}

          {/* Data table */}
          {isLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 6 }).map((_, i) => (
                <Skeleton key={i} className="h-8 w-full" />
              ))}
            </div>
          ) : dataset ? (
            <DataTable
              data={dataset.sample_data as Record<string, unknown>[]}
              columnInfo={dataset.columns}
              pageSize={20}
            />
          ) : null}

          {/* Continue to Explore */}
          <div className="flex justify-end pt-2">
            <Button
              size="lg"
              onClick={() => router.push("/explore")}
              disabled={!fileId}
              className="gap-2"
            >
              Continue to Explore
              <ArrowRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
