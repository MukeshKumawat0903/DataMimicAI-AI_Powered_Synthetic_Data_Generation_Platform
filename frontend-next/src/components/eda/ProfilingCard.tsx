"use client";

import type { ComponentProps } from "react";
import { useState } from "react";
import { useProfile } from "@/hooks/api/useProfile";
import { useCleaningAction, useDistributions, type CleaningKind } from "@/hooks/api/useAnalysis";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ChevronDown, ChevronRight, Lock, AlertTriangle, Loader2, Wrench, ScanSearch, Rows3, Columns3, SearchX, ListX, Hash, TrendingUp, Activity } from "lucide-react";
import { StatTile } from "@/components/ui/stat-tile";
import { TypeBadge } from "@/components/data/TypeBadge";
import { IconTile } from "@/components/ui/icon-tile";
import { toast } from "sonner";
import type { AppError, DistributionsResponse } from "@/lib/api/types";

// Profile suggestion fix_type → the cleaning action that fixes it ("review" needs a human)
const FIX: Record<string, { kind: CleaningKind; label: string }> = {
  impute: { kind: "impute", label: "Fill missing" },
  encode: { kind: "encode", label: "Group rare values" },
  outlier: { kind: "drop_outliers", label: "Drop outlier rows" },
  drop_constant: { kind: "drop_column", label: "Drop column" },
  drop_duplicate: { kind: "drop_column", label: "Drop column" },
  drop_high_missing: { kind: "drop_column", label: "Drop column" },
};

function Stat(props: ComponentProps<typeof StatTile>) {
  return <StatTile {...props} className="px-3 py-2.5" />;
}

export function ProfilingCard() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const [open, setOpen] = useState(false);
  const [triggered, setTriggered] = useState(false);
  const { data, isLoading, error, refetch } = useProfile(triggered ? fileId : null);
  const clean = useCleaningAction();
  const distributions = useDistributions();
  const [dist, setDist] = useState<DistributionsResponse | null>(null);

  function handleRun() {
    setTriggered(true);
    setOpen(true);
    if (triggered) refetch();
  }

  function applyFix(column: string, fixType: string, actionText: string) {
    const fix = FIX[fixType];
    if (!fileId || !fix) return;
    // The suggestion text names the imputation method ("... with median")
    const method = actionText.match(/with (mean|median)/)?.[1] ?? "median";
    clean.mutate(
      { fileId, kind: fix.kind, column, method },
      {
        onSuccess: () => toast.success(`${fix.label}: ${column}`, { description: "The dataset was updated." }),
        onError: (e) => toast.error(`${fix.label} failed`, { description: (e as Partial<AppError>).message }),
      }
    );
  }

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between py-3">
          <CollapsibleTrigger className="flex items-center gap-3 text-sm font-medium">
            {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            <IconTile icon={ScanSearch} tone="sky" />
            <span className="text-left">
              <CardTitle className="text-base">Data Profiling</CardTitle>
              <span className="block text-xs font-normal text-muted-foreground">Types, missing values, distributions and one-click fixes</span>
            </span>
          </CollapsibleTrigger>
          <Button size="sm" variant={triggered ? "outline" : "default"} onClick={handleRun}>
            {triggered ? "Re-run" : "Run Profile"}
          </Button>
        </CardHeader>

        <CollapsibleContent>
          <CardContent className="pt-0 space-y-4">
            {isLoading && (
              <div className="space-y-2">
                {[...Array(5)].map((_, i) => (
                  <Skeleton key={i} className="h-8 w-full" />
                ))}
              </div>
            )}
            {error && <p className="text-sm text-destructive">Failed to load profile. Please try again.</p>}

            {data && (
              <>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  <Stat label="Rows" value={data.summary.n_rows.toLocaleString()} icon={Rows3} tone="sky" />
                  <Stat label="Columns" value={data.summary.n_columns} icon={Columns3} tone="violet" />
                  <Stat label="Missing cells" value={data.summary.total_missing.toLocaleString()} icon={SearchX} tone="emerald" warn={data.summary.total_missing > 0} />
                  <Stat label="Columns with missing" value={data.summary.columns_with_missing} icon={ListX} tone="emerald" warn={data.summary.columns_with_missing > 0} />
                </div>
                {[
                  ["Constant columns", data.summary.constant_columns],
                  ["Duplicated columns", data.summary.duplicated_columns],
                  ["Mostly missing", data.summary.columns_high_missing],
                ].map(([label, cols]) =>
                  (cols as string[]).length ? (
                    <p key={label as string} className="text-xs text-amber-700">
                      {label}: <span className="font-mono">{(cols as string[]).join(", ")}</span>
                    </p>
                  ) : null
                )}

                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Column</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Missing %</TableHead>
                        <TableHead>Unique</TableHead>
                        <TableHead>Suggestions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {data.columns.map((col) => (
                        <TableRow key={col.column}>
                          <TableCell className="font-mono text-xs">
                            <span className="flex items-center gap-1">
                              {col.is_pii && <Lock className="h-3 w-3 text-amber-500" aria-label="PII column" />}
                              {col.column}
                            </span>
                          </TableCell>
                          <TableCell>
                            <TypeBadge type={col.dtype} />
                          </TableCell>
                          <TableCell>
                            <span
                              className={
                                col.missing_pct > 20
                                  ? "text-destructive font-medium"
                                  : col.missing_pct > 5
                                  ? "text-amber-600 font-medium"
                                  : ""
                              }
                            >
                              {col.missing_pct.toFixed(1)}%
                            </span>
                          </TableCell>
                          <TableCell>{col.unique_count}</TableCell>
                          <TableCell>
                            <div className="flex flex-col gap-1">
                              {col.fixes.map((f) => (
                                <div key={f.action} className="flex flex-wrap items-center gap-1.5">
                                  {/* Sentences wrap, so not a fixed-height badge */}
                                  <span className="rounded-md bg-amber-50 px-2 py-1 text-xs leading-snug text-amber-800 ring-1 ring-inset ring-amber-600/15 dark:bg-amber-500/10 dark:text-amber-200">
                                    {f.action}
                                  </span>
                                  {FIX[f.fix_type] && (
                                    <Button
                                      size="sm"
                                      variant="outline"
                                      className="h-6 gap-1 px-2 text-xs"
                                      disabled={clean.isPending}
                                      onClick={() => applyFix(col.column, f.fix_type, f.action)}
                                    >
                                      <Wrench className="h-3 w-3" />
                                      {FIX[f.fix_type].label}
                                    </Button>
                                  )}
                                </div>
                              ))}
                              {col.missing_pct > 20 && <AlertTriangle className="h-3.5 w-3.5 text-amber-500" />}
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                  <p className="mt-2 text-xs text-muted-foreground">
                    {data.row_count.toLocaleString()} rows analysed. Fixes change the working dataset; use
                    &ldquo;Reset to original&rdquo; at the top of the page to undo them.
                  </p>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-sm font-medium">Distribution analysis</p>
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={distributions.isPending}
                      onClick={() =>
                        fileId &&
                        distributions.mutate(fileId, {
                          onSuccess: setDist,
                          onError: (e) => toast.error("Distribution fitting failed", { description: (e as Partial<AppError>).message }),
                        })
                      }
                    >
                      {distributions.isPending && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1" />}
                      Fit distributions
                    </Button>
                  </div>
                  {dist && (
                    <>
                      <div className="grid grid-cols-3 gap-2">
                        <Stat label="Numeric columns" value={dist.summary.total_columns_analyzed} icon={Hash} tone="sky" />
                        <Stat label="Highly skewed" value={dist.summary.highly_skewed_columns.length} icon={TrendingUp} tone="orange" />
                        <Stat label="Heavy-tailed" value={dist.summary.heavy_tailed_columns.length} icon={Activity} tone="pink" />
                      </div>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Column</TableHead>
                            <TableHead>Best fit</TableHead>
                            <TableHead className="text-right">Skewness</TableHead>
                            <TableHead className="text-right">Kurtosis</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {Object.entries(dist.distributions).map(([col, d]) => (
                            <TableRow key={col}>
                              <TableCell className="font-mono text-xs">{col}</TableCell>
                              <TableCell>{d.error ? <span className="text-destructive text-xs">failed</span> : d.best_distribution}</TableCell>
                              <TableCell className="text-right">{d.skewness?.toFixed(2) ?? "—"}</TableCell>
                              <TableCell className="text-right">{d.kurtosis?.toFixed(2) ?? "—"}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                      {dist.summary.recommendations.length > 0 && (
                        <ul className="list-disc pl-5 text-xs text-muted-foreground space-y-0.5">
                          {dist.summary.recommendations.map((r) => (
                            <li key={r}>{r}</li>
                          ))}
                        </ul>
                      )}
                    </>
                  )}
                </div>
              </>
            )}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
