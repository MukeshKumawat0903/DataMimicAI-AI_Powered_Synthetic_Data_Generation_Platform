"use client";

import { useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { StatTile } from "@/components/ui/stat-tile";
import { IconTile } from "@/components/ui/icon-tile";
import { TypeBadge } from "@/components/data/TypeBadge";
import { AlertTriangle, Columns3, Copy, Gauge, Info, Rows3, SearchX } from "lucide-react";
import { cn } from "@/lib/utils";
import type { DatasetResponse } from "@/lib/api/types";

interface Overview {
  missingCells: number;
  missingPct: number;
  duplicateRows: number;
  duplicatePct: number;
  constantColumns: string[];
  typeCounts: Record<string, number>;
  score: number;
}

/** Same scoring as the Streamlit Smart Preview: start at 100 and deduct for each problem */
function computeOverview(ds: DatasetResponse): Overview {
  const rows = ds.sample_data;
  const names = ds.columns.map((c) => c.name);
  let missingCells = 0;
  const seen = new Set<string>();
  let duplicateRows = 0;
  for (const row of rows) {
    for (const n of names) if (row[n] === null || row[n] === undefined) missingCells++;
    const key = JSON.stringify(names.map((n) => row[n]));
    if (seen.has(key)) duplicateRows++;
    else seen.add(key);
  }
  const cells = rows.length * names.length;
  const missingPct = cells ? (missingCells / cells) * 100 : 0;
  const duplicatePct = rows.length ? (duplicateRows / rows.length) * 100 : 0;
  const constantColumns = ds.columns
    .filter((c) => (c.type === "categorical" || c.type === "text") && c.unique_count === 1)
    .map((c) => c.name);
  const typeCounts: Record<string, number> = {};
  for (const c of ds.columns) typeCounts[c.type] = (typeCounts[c.type] ?? 0) + 1;

  let score = 100;
  if (missingCells) score -= Math.min(missingPct * 2, 30);
  if (duplicateRows) score -= Math.min(duplicatePct, 20);
  score -= constantColumns.length * 5;
  return {
    missingCells,
    missingPct,
    duplicateRows,
    duplicatePct,
    constantColumns,
    typeCounts,
    score: Math.round(Math.max(0, Math.min(100, score))),
  };
}

function scoreBand(score: number) {
  if (score >= 90) return { label: "Excellent", bar: "from-emerald-400 to-teal-500", text: "text-emerald-600 dark:text-emerald-400" };
  if (score >= 70) return { label: "Good", bar: "from-sky-400 to-indigo-500", text: "text-sky-600 dark:text-sky-400" };
  if (score >= 50) return { label: "Fair", bar: "from-amber-400 to-orange-500", text: "text-amber-600 dark:text-amber-400" };
  return { label: "Needs attention", bar: "from-rose-400 to-red-500", text: "text-red-600 dark:text-red-400" };
}

export function DataOverview({ dataset }: { dataset: DatasetResponse }) {
  const o = useMemo(() => computeOverview(dataset), [dataset]);
  const band = scoreBand(o.score);
  const issues: { text: string; severe: boolean }[] = [];
  if (o.missingCells)
    issues.push({
      text: `${o.missingCells.toLocaleString()} missing cells (${o.missingPct.toFixed(2)}%)`,
      severe: o.missingPct >= 20,
    });
  if (o.duplicateRows)
    issues.push({
      text: `${o.duplicateRows.toLocaleString()} duplicate rows (${o.duplicatePct.toFixed(2)}%)`,
      severe: o.duplicatePct >= 5,
    });
  if (o.constantColumns.length)
    issues.push({ text: `Constant columns: ${o.constantColumns.join(", ")}`, severe: false });

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile label="Rows" value={dataset.num_rows.toLocaleString()} icon={Rows3} tone="sky" />
        <StatTile label="Columns" value={String(dataset.num_columns)} icon={Columns3} tone="violet" />
        <StatTile
          label="Missing values"
          value={o.missingCells.toLocaleString()}
          sub={`${o.missingPct.toFixed(2)}% of cells`}
          icon={SearchX}
          tone="emerald"
          warn={o.missingCells > 0}
        />
        <StatTile
          label="Duplicates"
          value={o.duplicateRows.toLocaleString()}
          sub={`${o.duplicatePct.toFixed(2)}% of rows`}
          icon={Copy}
          tone="emerald"
          warn={o.duplicateRows > 0}
        />
      </div>

      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-muted-foreground">Column types:</span>
        {Object.entries(o.typeCounts).map(([type, n]) => (
          <TypeBadge key={type} type={`${type}: ${n}`} />
        ))}
      </div>

      <Card>
        <CardContent className="space-y-2.5 px-4 py-1">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <IconTile icon={Gauge} tone="indigo" />
              <p className="text-sm font-medium">
                Data quality score: <span className={cn("text-base font-bold", band.text)}>{o.score}/100</span>
              </p>
            </div>
            <span className={cn("rounded-full bg-muted px-2.5 py-0.5 text-xs font-semibold", band.text)}>{band.label}</span>
          </div>
          <div className="h-2.5 rounded-full bg-muted overflow-hidden" role="progressbar" aria-valuenow={o.score} aria-valuemin={0} aria-valuemax={100}>
            <div className={cn("h-full rounded-full bg-gradient-to-r transition-all duration-700", band.bar)} style={{ width: `${o.score}%` }} />
          </div>
          {issues.length === 0 ? (
            <p className="text-xs text-muted-foreground">No missing values, duplicate rows or constant columns.</p>
          ) : (
            <ul className="space-y-1">
              {issues.map((i) => (
                <li key={i.text} className="flex items-start gap-1.5 text-xs">
                  {i.severe ? (
                    <AlertTriangle className="h-3.5 w-3.5 text-destructive shrink-0" />
                  ) : (
                    <Info className="h-3.5 w-3.5 text-amber-500 shrink-0" />
                  )}
                  {i.text}
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
