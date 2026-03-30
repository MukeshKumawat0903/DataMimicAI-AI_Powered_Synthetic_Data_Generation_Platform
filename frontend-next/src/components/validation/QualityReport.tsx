"use client";

import { useRouter } from "next/navigation";
import { useQuality } from "@/hooks/api/useQuality";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { GaugeChart, ScoreLegend } from "@/components/charts/GaugeChart";
import { EmptyState } from "@/components/feedback/EmptyState";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertTriangle, BarChart2, CheckCircle2, Info, RefreshCw } from "lucide-react";

const SEVERITY_ICON = {
  high: <AlertTriangle className="h-3.5 w-3.5 text-destructive shrink-0 mt-0.5" />,
  medium: <AlertTriangle className="h-3.5 w-3.5 text-amber-500 shrink-0 mt-0.5" />,
  low: <Info className="h-3.5 w-3.5 text-blue-500 shrink-0 mt-0.5" />,
};

export function QualityReport() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const generatedFileId = useWorkspaceStore((s) => s.generatedFileId);
  const { data, isLoading, error, refetch } = useQuality(fileId);
  const router = useRouter();

  // P1.2: Show empty state with CTA when no synthetic data has been generated yet
  if (!generatedFileId) {
    return (
      <Card>
        <CardHeader><CardTitle>Quality Report</CardTitle></CardHeader>
        <CardContent>
          <EmptyState
            icon={<BarChart2 className="h-8 w-8" />}
            title="No synthetic data yet"
            description="Generate a synthetic dataset to compute fidelity, privacy, and utility scores."
            action={{ label: "Go to Generate", onClick: () => router.push("/generate") }}
          />
        </CardContent>
      </Card>
    );
  }

  if (isLoading) {
    return (
      <Card>
        <CardHeader><CardTitle>Quality Report</CardTitle></CardHeader>
        {/* P3.2: aria-live so screen readers announce when scores load */}
        <CardContent aria-live="polite" aria-atomic="true" className="space-y-4">
          <div className="flex gap-8 justify-center">
            {[...Array(3)].map((_, i) => (
              <Skeleton key={i} className="h-32 w-32 rounded-full" />
            ))}
          </div>
          <Skeleton className="h-16 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (error || !data) {
    return (
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Quality Report</CardTitle>
          <Button size="sm" variant="outline" onClick={() => refetch()}>
            <RefreshCw className="h-3.5 w-3.5 mr-1" />
            Retry
          </Button>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-destructive">
            Failed to compute quality metrics. Generate synthetic data first, or check backend connection.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Quality Report</CardTitle>
        <Button size="sm" variant="outline" onClick={() => refetch()}>
          <RefreshCw className="h-3.5 w-3.5 mr-1" />
          Refresh
        </Button>
      </CardHeader>
      {/* P3.2: aria-live so screen readers announce score updates */}
      <CardContent aria-live="polite" aria-atomic="true" className="space-y-6">
        {/* Gauge charts */}
        <div className="flex gap-8 justify-center flex-wrap">
          <GaugeChart value={data.scores.fidelity} label="Fidelity" />
          <GaugeChart value={data.scores.privacy} label="Privacy" />
          <GaugeChart value={data.scores.utility} label="Utility" />
        </div>
        <div className="flex justify-center">
          <ScoreLegend />
        </div>

        {/* Issues */}
        {data.issues.length > 0 && (
          <div className="space-y-2">
            <p className="text-sm font-medium">Issues</p>
            {data.issues.map((issue, i) => (
              <div key={i} className="flex items-start gap-2 text-sm">
                {SEVERITY_ICON[issue.severity]}
                <span>{issue.message}</span>
                <Badge variant="outline" className="text-xs shrink-0 ml-auto">
                  {issue.category}
                </Badge>
              </div>
            ))}
          </div>
        )}

        {/* Recommendation */}
        <div className="rounded-lg border border-blue-200 bg-blue-50 dark:bg-blue-950/20 dark:border-blue-800 p-3 flex items-start gap-2">
          <CheckCircle2 className="h-4 w-4 text-blue-500 shrink-0 mt-0.5" />
          <div>
            <p className="text-xs font-medium text-blue-700 dark:text-blue-400">
              Recommendation — Weakest: {data.weakest_category}
            </p>
            <p className="text-sm text-blue-700 dark:text-blue-300 mt-0.5">
              {data.recommendation}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
