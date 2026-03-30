"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { QualityReport } from "@/components/validation/QualityReport";
import { VizDashboard } from "@/components/validation/VizDashboard";
import { RefinementPanel } from "@/components/validation/RefinementPanel";
import { VersionTimeline } from "@/components/validation/VersionTimeline";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Download } from "lucide-react";
import Link from "next/link";
import { apiClient } from "@/lib/api/client";
import { toast } from "sonner";

async function downloadFile(fileId: string) {
  const res = await apiClient.get(`/eda/download`, {
    params: { file_id: `syn_${fileId}` },
    responseType: "blob",
  });
  const url = URL.createObjectURL(res.data as Blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `synthetic_data.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export default function ValidatePage() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const generatedFileId = useWorkspaceStore((s) => s.generatedFileId);
  const router = useRouter();

  useEffect(() => {
    if (!fileId) {
      router.replace("/upload");
    }
  }, [fileId, router]);

  if (!fileId) return null;

  return (
    <div className="max-w-5xl mx-auto space-y-4 p-4">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-2xl font-semibold">Validate &amp; Refine</h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            Inspect quality scores and visualize differences between real and synthetic data.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link href="/generate">
            <Button variant="ghost" size="sm" className="gap-1">
              <ArrowLeft className="h-4 w-4" />
              Back
            </Button>
          </Link>
          {generatedFileId && (
            <>
              <Button
                variant="outline"
                size="sm"
                className="gap-1"
                onClick={() => {
                  // P0.2: Toast on download
                  downloadFile(fileId)
                    .then(() => toast.success("Download started", { description: "synthetic_data.csv" }))
                    .catch(() => toast.error("Download failed", { description: "Please try again." }));
                }}
              >
                <Download className="h-3.5 w-3.5" />
                Download CSV
              </Button>
            </>
          )}
        </div>
      </div>

      {!generatedFileId && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 dark:bg-amber-950/20 dark:border-amber-800 px-4 py-3 text-sm text-amber-700 dark:text-amber-400">
          No synthetic data found. Please generate data first.{" "}
          <Link href="/generate" className="underline font-medium">
            Go to Generate →
          </Link>
        </div>
      )}

      <Tabs defaultValue="quality">
        <TabsList>
          <TabsTrigger value="quality">Quality Report</TabsTrigger>
          <TabsTrigger value="visualize">Visualizations</TabsTrigger>
          <TabsTrigger value="iterate">Iterate</TabsTrigger>
        </TabsList>

        <TabsContent value="quality" className="mt-4">
          <QualityReport />
        </TabsContent>

        <TabsContent value="visualize" className="mt-4">
          {generatedFileId ? (
            <VizDashboard />
          ) : (
            <p className="text-sm text-muted-foreground">
              Visualizations available after generating synthetic data.
            </p>
          )}
        </TabsContent>

        <TabsContent value="iterate" className="mt-4 space-y-4">
          <RefinementPanel />
          <VersionTimeline />
        </TabsContent>
      </Tabs>
    </div>
  );
}
