"use client";

import { useMemo, useState } from "react";
import { useDecisionReport, useValidationPlans } from "@/hooks/api/useAgents";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Label } from "@/components/ui/label";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { cn, humanize } from "@/lib/utils";
import type { AppError, DecisionMetric } from "@/lib/api/types";
import { FileBarChart } from "lucide-react";
import { IconTile } from "@/components/ui/icon-tile";

type Change = "Improved" | "Worsened" | "Changed" | "Unchanged" | "Unavailable";

// Metrics where a smaller value is better; skewness is better closer to zero
const LOWER_IS_BETTER = new Set(["missing_pct", "outlier_pct", "overall_missing_pct"]);

function classify(m: DecisionMetric): Change {
  if (m.before == null || m.after == null || m.delta == null) return "Unavailable";
  if (Math.abs(m.delta) <= 0.001) return "Unchanged";
  if (m.metric === "skewness") return Math.abs(m.after) < Math.abs(m.before) ? "Improved" : "Worsened";
  if (LOWER_IS_BETTER.has(m.metric)) return m.after < m.before ? "Improved" : "Worsened";
  return "Changed";
}

const CHANGE_CLASS: Record<Change, string> = {
  Improved: "text-green-600",
  Worsened: "text-destructive",
  Changed: "text-amber-600",
  Unchanged: "text-muted-foreground",
  Unavailable: "text-muted-foreground",
};

const fmt = (v: number | null) => (v == null ? "—" : Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3));

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border p-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-lg font-semibold">{value}</p>
    </div>
  );
}

export function DecisionReport({
  planId,
  onPlanChange,
}: {
  planId: string | null;
  onPlanChange: (planId: string) => void;
}) {
  const { data: planIds, isLoading: plansLoading } = useValidationPlans();
  const selected = planId ?? planIds?.[planIds.length - 1] ?? null;
  const { data: report, isLoading, error } = useDecisionReport(selected);
  const [changeFilter, setChangeFilter] = useState<"All" | "Changed" | "Unchanged">("All");
  const [columnFilter, setColumnFilter] = useState("All");

  const rows = useMemo(
    () => (report?.validation_results ?? []).map((m) => ({ ...m, change: classify(m) })),
    [report]
  );
  const changed = rows.filter((r) => r.change !== "Unchanged" && r.change !== "Unavailable");
  const improved = rows.filter((r) => r.change === "Improved").length;
  const worsened = rows.filter((r) => r.change === "Worsened").length;
  const topChanges = [...changed].sort((a, b) => Math.abs(b.delta ?? 0) - Math.abs(a.delta ?? 0)).slice(0, 10);
  const columns = Array.from(new Set(rows.map((r) => r.column)));
  const visible = rows.filter(
    (r) =>
      (changeFilter === "All" || (changeFilter === "Changed" ? r.change !== "Unchanged" : r.change === "Unchanged")) &&
      (columnFilter === "All" || r.column === columnFilter)
  );

  return (
    <Card>
      <CardHeader className="flex flex-row items-start gap-3 py-3">
        <IconTile icon={FileBarChart} tone="sky" />
        <div className="space-y-1">
          <CardTitle className="text-base">Decision Report</CardTitle>
          <p className="text-sm text-muted-foreground">
            What an executed plan changed: each metric measured on the data before and after the
            transformations.
          </p>
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        {plansLoading ? (
          <Skeleton className="h-9 w-48" />
        ) : !planIds?.length ? (
          <p className="text-sm text-muted-foreground">
            No executed plans yet. Approve and execute a plan in the Action Planner first.
          </p>
        ) : (
          <div className="space-y-1.5 max-w-xs">
            <Label htmlFor="report-plan">Plan</Label>
            <Select value={selected ?? undefined} onValueChange={(v) => onPlanChange(v as string)}>
              <SelectTrigger id="report-plan" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {planIds.map((id) => (
                  <SelectItem key={id} value={id}>
                    {id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        {isLoading && <Skeleton className="h-40 w-full" />}
        {error && (
          <p className="text-sm text-destructive">
            Couldn&apos;t load the report: {(error as Partial<AppError>).message}
          </p>
        )}

        {report && (
          <>
            <div className="grid gap-3 grid-cols-2 md:grid-cols-4">
              <Stat label="Metrics compared" value={`${report.summary.metrics_compared} / ${report.summary.total_metrics}`} />
              <Stat label="Columns affected" value={String(report.summary.columns_affected)} />
              <Stat
                label="Rows (before → after)"
                value={`${report.summary.original_shape.rows} → ${report.summary.transformed_shape.rows}`}
              />
              <Stat
                label="Columns (before → after)"
                value={`${report.summary.original_shape.columns} → ${report.summary.transformed_shape.columns}`}
              />
            </div>

            <p className="text-sm">
              {changed.length} of {rows.length} metrics changed:{" "}
              <span className="text-green-600 font-medium">{improved} improved</span>,{" "}
              <span className="text-destructive font-medium">{worsened} worsened</span>.
              {topChanges[0] && (
                <>
                  {" "}Biggest change: {humanize(topChanges[0].metric)} of{" "}
                  <span className="font-mono">{topChanges[0].column}</span> went from {fmt(topChanges[0].before)} to{" "}
                  {fmt(topChanges[0].after)}.
                </>
              )}
            </p>

            {topChanges.length > 0 && (
              <div className="h-64" aria-label="Before and after values for the largest changes">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={topChanges.map((r) => ({ name: `${r.column} · ${humanize(r.metric)}`, before: r.before, after: r.after }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" fontSize={10} interval={0} angle={-20} textAnchor="end" height={70} />
                    <YAxis fontSize={11} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="before" name="Before" fill="#94a3b8" />
                    <Bar dataKey="after" name="After" fill="#2563eb" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            <div className="space-y-2">
              <div className="flex flex-wrap items-end gap-3">
                <div className="space-y-1">
                  <Label htmlFor="report-change">Show</Label>
                  <Select value={changeFilter} onValueChange={(v) => setChangeFilter(v as typeof changeFilter)}>
                    <SelectTrigger id="report-change" className="w-36">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {["All", "Changed", "Unchanged"].map((v) => (
                        <SelectItem key={v} value={v}>
                          {v}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="report-column">Column</Label>
                  <Select value={columnFilter} onValueChange={(v) => setColumnFilter(v as string)}>
                    <SelectTrigger id="report-column" className="w-56">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {["All", ...columns].map((c) => (
                        <SelectItem key={c} value={c}>
                          {c}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Column</TableHead>
                    <TableHead>Metric</TableHead>
                    <TableHead className="text-right">Before</TableHead>
                    <TableHead className="text-right">After</TableHead>
                    <TableHead className="text-right">Change</TableHead>
                    <TableHead>Result</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {visible.map((r, i) => (
                    <TableRow key={i}>
                      <TableCell className="font-mono text-xs">{r.column}</TableCell>
                      <TableCell>{humanize(r.metric)}</TableCell>
                      <TableCell className="text-right">{fmt(r.before)}</TableCell>
                      <TableCell className="text-right">{fmt(r.after)}</TableCell>
                      <TableCell className="text-right">{r.delta == null ? "—" : (r.delta > 0 ? "+" : "") + fmt(r.delta)}</TableCell>
                      <TableCell className={cn("text-sm", CHANGE_CLASS[r.change])}>{r.change}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
