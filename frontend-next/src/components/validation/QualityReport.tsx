"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useQuality } from "@/hooks/api/useQuality";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { GaugeChart, ScoreLegend } from "@/components/charts/GaugeChart";
import { EmptyState } from "@/components/feedback/EmptyState";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { AlertTriangle, BarChart2, Info, RefreshCw, Gauge } from "lucide-react";
import { IconTile } from "@/components/ui/icon-tile";
import type { TableSummary } from "@/lib/api/types";
import { QualityActions } from "./QualityActions";

const SEVERITY_ICON = {
  high: <AlertTriangle className="h-3.5 w-3.5 text-destructive shrink-0 mt-0.5" />,
  medium: <AlertTriangle className="h-3.5 w-3.5 text-amber-500 shrink-0 mt-0.5" />,
  low: <Info className="h-3.5 w-3.5 text-blue-500 shrink-0 mt-0.5" />,
};

type MatchStatus = "Match" | "Better" | "Worse" | "Different" | "Set by you";

const STATUS_CLASS: Record<MatchStatus, string> = {
  Match: "text-green-600",
  Better: "text-green-600",
  "Set by you": "text-muted-foreground",
  Worse: "text-destructive",
  Different: "text-amber-600",
};

/** Rows for the real-vs-synthetic table; lower is better for missing values and duplicates */
function keyMetricRows(real: TableSummary, synth: TableSummary) {
  const lowerIsBetter = (r: number, s: number): MatchStatus => (s === r ? "Match" : s < r ? "Better" : "Worse");
  const same = (r: number, s: number): MatchStatus => (s === r ? "Match" : "Different");
  return [
    { metric: "Row count", real: real.rows.toLocaleString(), synth: synth.rows.toLocaleString(),
      status: (synth.rows === real.rows ? "Match" : "Set by you") as MatchStatus },
    { metric: "Column count", real: real.columns, synth: synth.columns, status: same(real.columns, synth.columns) },
    { metric: "Missing values", real: `${real.missing_pct}%`, synth: `${synth.missing_pct}%`,
      status: lowerIsBetter(real.missing_pct, synth.missing_pct) },
    { metric: "Duplicate rows", real: real.duplicate_rows, synth: synth.duplicate_rows,
      status: lowerIsBetter(real.duplicate_rows, synth.duplicate_rows) },
    { metric: "Numeric columns", real: real.numeric_columns, synth: synth.numeric_columns,
      status: same(real.numeric_columns, synth.numeric_columns) },
    { metric: "Categorical columns", real: real.categorical_columns, synth: synth.categorical_columns,
      status: same(real.categorical_columns, synth.categorical_columns) },
  ];
}

export function QualityReport() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const generatedFileId = useWorkspaceStore((s) => s.generatedFileId);
  const setLatestGenerationScores = useWorkspaceStore((s) => s.setLatestGenerationScores);
  const { data, isLoading, error, refetch } = useQuality(fileId);
  const router = useRouter();

  // Keep the scores with the generation they belong to (for the version history)
  useEffect(() => {
    if (data) setLatestGenerationScores(data.scores);
  }, [data, setLatestGenerationScores]);

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
        <div className="flex items-center gap-3">
          <IconTile icon={Gauge} tone="indigo" />
          <div>
            <CardTitle>Quality Report</CardTitle>
            <p className="text-xs text-muted-foreground">Fidelity, privacy and utility of the latest generation</p>
          </div>
        </div>
        <Button size="sm" variant="outline" onClick={() => refetch()}>
          <RefreshCw className="h-3.5 w-3.5 mr-1" />
          Refresh
        </Button>
      </CardHeader>
      {/* P3.2: aria-live so screen readers announce score updates */}
      <CardContent aria-live="polite" aria-atomic="true" className="space-y-6">
        {/* Gauge charts */}
        <div className="grid gap-3 sm:grid-cols-3">
          {([["Fidelity", data.scores.fidelity], ["Privacy", data.scores.privacy], ["Utility", data.scores.utility]] as const).map(([label, value]) => (
            <div key={label} className="flex justify-center rounded-2xl border bg-gradient-to-b from-muted/50 to-card py-4">
              <GaugeChart value={value} label={label} size={132} />
            </div>
          ))}
        </div>
        <div className="flex justify-center">
          <ScoreLegend />
        </div>
        <p className="text-xs text-muted-foreground text-center">
          {data.privacy_detail
            ? `Privacy = how far synthetic rows are from their closest real record, compared with random data (${data.privacy_detail.dcr_baseline_protection.toFixed(2)}). ${Math.round(data.privacy_detail.new_row_synthesis * 100)}% of synthetic rows are new.`
            : "Privacy metrics are unavailable for this dataset."}
        </p>

        {/* Key metrics: real vs synthetic */}
        {data.summary && (
          <div className="space-y-2">
            <p className="text-sm font-medium">Key Metrics: Real vs Synthetic</p>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Metric</TableHead>
                  <TableHead>Real</TableHead>
                  <TableHead>Synthetic</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {keyMetricRows(data.summary.real, data.summary.synthetic).map((row) => (
                  <TableRow key={row.metric}>
                    <TableCell>{row.metric}</TableCell>
                    <TableCell>{row.real}</TableCell>
                    <TableCell>{row.synth}</TableCell>
                    <TableCell className={STATUS_CLASS[row.status]}>{row.status}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}

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

        {/* Recommendations (with one-click apply) and exports */}
        <QualityActions data={data} />
      </CardContent>
    </Card>
  );
}
