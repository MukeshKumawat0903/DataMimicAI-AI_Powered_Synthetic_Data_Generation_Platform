"use client";

import type { ReactNode } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { IconTile, type Tone } from "@/components/ui/icon-tile";
import { ArrowRight, BarChart2, FlaskConical, ShieldCheck, Table2, Upload, Wand2, type LucideIcon } from "lucide-react";
import { FileUploader } from "@/components/data/FileUploader";
import { DemoModeSelector } from "@/components/data/DemoModeSelector";
import { DataTable } from "@/components/data/DataTable";
import { KPIBanner } from "@/components/data/KPIBanner";
import { DataOverview } from "@/components/data/DataOverview";
import { PageHeader } from "@/components/layout/PageHeader";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { useDataset } from "@/hooks/api/useDataset";

const HOW_IT_WORKS: { icon: LucideIcon; tone: Tone; title: string; text: string }[] = [
  { icon: Upload, tone: "violet", title: "Upload", text: "Bring a CSV, or start from a demo dataset." },
  { icon: BarChart2, tone: "sky", title: "Explore", text: "Profile, clean and spot privacy risks, with AI help." },
  { icon: Wand2, tone: "pink", title: "Generate", text: "Train SDV or SynthCity models on your data." },
  { icon: ShieldCheck, tone: "emerald", title: "Validate", text: "Check fidelity, utility and privacy, then refine." },
];

function SectionLabel({ icon: Icon, children }: { icon: LucideIcon; children: ReactNode }) {
  return (
    <h2 className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
      <Icon className="h-3.5 w-3.5 text-primary" />
      {children}
    </h2>
  );
}

export default function UploadPage() {
  const router = useRouter();
  const fileId = useWorkspaceStore((s) => s.fileId);
  const kpiSummary = useWorkspaceStore((s) => s.kpiSummary);

  const { data: dataset, isLoading } = useDataset(fileId);

  return (
    <div className="mx-auto max-w-6xl space-y-8 px-6 py-8">
      <PageHeader
        icon={Upload}
        step="Step 1 of 4"
        title="Upload Dataset"
        description="Upload a CSV file or load a demo dataset to get started."
      />

      {/* Upload / Demo section */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_auto_minmax(0,20rem)]">
        <div>
          <SectionLabel icon={Upload}>Upload your file</SectionLabel>
          <FileUploader />
        </div>

        {/* Divider */}
        <div className="hidden lg:flex flex-col items-center gap-2 pt-8">
          <span className="w-px flex-1 bg-gradient-to-b from-transparent via-border to-transparent" />
          <span className="rounded-full border bg-background px-2.5 py-1 text-[11px] font-semibold text-muted-foreground shadow-sm">
            OR
          </span>
          <span className="w-px flex-1 bg-gradient-to-b from-transparent via-border to-transparent" />
        </div>

        <div>
          <SectionLabel icon={FlaskConical}>Use demo data</SectionLabel>
          <DemoModeSelector />
        </div>
      </div>

      {/* First visit: what the four steps do */}
      {!fileId && (
        <div>
          <SectionLabel icon={ArrowRight}>How it works</SectionLabel>
          <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {HOW_IT_WORKS.map((s, i) => (
              <li key={s.title} className="relative rounded-2xl border bg-card p-4 shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md hover:shadow-indigo-500/10">
                <div className="flex items-center gap-3">
                  <IconTile icon={s.icon} tone={s.tone} />
                  <span className="text-xs font-semibold text-muted-foreground">Step {i + 1}</span>
                </div>
                <p className="mt-3 font-semibold">{s.title}</p>
                <p className="mt-0.5 text-sm text-muted-foreground">{s.text}</p>
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* Data preview section — shown once a dataset is loaded */}
      {fileId && (
        <div className="space-y-5">
          <div className="flex items-center gap-3 border-t pt-6">
            <IconTile icon={Table2} tone="sky" />
            <div>
              <h2 className="text-lg font-semibold tracking-tight">Data Preview</h2>
              {dataset && (
                <p className="text-sm text-muted-foreground">
                  {dataset.num_rows.toLocaleString()} rows ×{" "}
                  {dataset.num_columns} columns
                </p>
              )}
            </div>
          </div>

          {/* Quick data overview + quality score */}
          {dataset && <DataOverview dataset={dataset} />}

          {/* Transformation impact (after previewing accepted suggestions in Explore) */}
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
