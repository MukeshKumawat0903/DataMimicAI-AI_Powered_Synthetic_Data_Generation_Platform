"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useStoreHydrated } from "@/lib/store/useStoreHydrated";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { QualityReport } from "@/components/validation/QualityReport";
import { VizDashboard } from "@/components/validation/VizDashboard";
import { RefinementPanel } from "@/components/validation/RefinementPanel";
import { VersionTimeline } from "@/components/validation/VersionTimeline";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ShieldCheck, Gauge, LineChart, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import Link from "next/link";
import { DownloadFormats } from "@/components/generation/DownloadFormats";

export default function ValidatePage() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const generatedFileId = useWorkspaceStore((s) => s.generatedFileId);
  const router = useRouter();

  // Wait for the saved session to load before deciding there's no dataset
  const hydrated = useStoreHydrated();
  useEffect(() => {
    if (hydrated && !fileId) {
      router.replace("/upload");
    }
  }, [hydrated, fileId, router]);

  if (!fileId) return null;

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-6 py-8">
      <PageHeader
        icon={ShieldCheck}
        step="Step 4 of 4"
        title="Validate & Refine"
        description="Inspect quality scores and visualize differences between real and synthetic data."
        actions={
          <>
            <Link href="/generate">
              <Button variant="ghost" size="sm" className="gap-1">
                <ArrowLeft className="h-4 w-4" />
                Back
              </Button>
            </Link>
            {generatedFileId && fileId && <DownloadFormats fileId={fileId} />}
          </>
        }
      />

      {!generatedFileId && (
        <div className="rounded-xl border border-amber-300 bg-amber-50 dark:bg-amber-500/10 dark:border-amber-500/30 px-4 py-3 text-sm text-amber-700 dark:text-amber-300">
          No synthetic data found. Please generate data first.{" "}
          <Link href="/generate" className="underline font-medium">
            Go to Generate →
          </Link>
        </div>
      )}

      <Tabs defaultValue="quality">
        <TabsList className="h-11 p-1">
          <TabsTrigger value="quality" className="gap-2 px-4">
            <Gauge className="h-4 w-4 text-indigo-500" />
            Quality Report
          </TabsTrigger>
          <TabsTrigger value="visualize" className="gap-2 px-4">
            <LineChart className="h-4 w-4 text-sky-500" />
            Visualizations
          </TabsTrigger>
          <TabsTrigger value="iterate" className="gap-2 px-4">
            <RefreshCw className="h-4 w-4 text-fuchsia-500" />
            Iterate
          </TabsTrigger>
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
