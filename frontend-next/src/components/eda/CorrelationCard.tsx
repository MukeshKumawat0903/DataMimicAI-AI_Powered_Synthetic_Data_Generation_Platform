"use client";

import { useState } from "react";
import { useCorrelation } from "@/hooks/api/useCorrelation";
import { useDependencies, useMutualInfo } from "@/hooks/api/useAnalysis";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ChevronDown, ChevronRight, AlertTriangle, Loader2, Network } from "lucide-react";
import { IconTile } from "@/components/ui/icon-tile";
import { toast } from "sonner";
import type { AppError, CorrelationMethod, DependenciesResponse, MutualInfoResponse } from "@/lib/api/types";

function correlationColor(v: number): string {
  const abs = Math.abs(v);
  if (abs > 0.99) return "text-destructive font-bold";
  if (abs > 0.8) return "text-red-600 font-medium";
  if (abs > 0.5) return "text-amber-600";
  return "";
}

/** Correlation matrix as a colour-coded table: red = positive, blue = negative */
function MatrixTable({ matrix }: { matrix: Record<string, Record<string, number>> }) {
  const cols = Object.keys(matrix);
  return (
    <div className="overflow-x-auto">
      <table className="text-xs border-collapse">
        <thead>
          <tr>
            <th />
            {cols.map((c) => (
              <th key={c} className="px-2 py-1 font-mono font-normal text-muted-foreground max-w-[110px] truncate" title={c}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cols.map((r) => (
            <tr key={r}>
              <th className="px-2 py-1 text-left font-mono font-normal text-muted-foreground max-w-[160px] truncate" title={r}>
                {r}
              </th>
              {cols.map((c) => {
                const v = matrix[r]?.[c] ?? 0;
                const bg = v >= 0 ? `rgba(220, 38, 38, ${Math.abs(v) * 0.6})` : `rgba(37, 99, 235, ${Math.abs(v) * 0.6})`;
                return (
                  <td key={c} className="px-2 py-1 text-center tabular-nums border" style={{ backgroundColor: bg }}>
                    {v.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function CorrelationCard() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const [open, setOpen] = useState(false);
  const [triggered, setTriggered] = useState(false);
  const [method, setMethod] = useState<CorrelationMethod>("pearson");
  const [topK, setTopK] = useState(10);
  const { data, isLoading, error, refetch } = useCorrelation(triggered ? fileId : null, topK);
  const mi = useMutualInfo();
  const deps = useDependencies();
  const [miResult, setMiResult] = useState<MutualInfoResponse | null>(null);
  const [depResult, setDepResult] = useState<DependenciesResponse | null>(null);

  function handleRun() {
    setTriggered(true);
    setOpen(true);
    if (triggered) refetch();
  }

  const failed = (what: string) => (e: unknown) => toast.error(`${what} failed`, { description: (e as Partial<AppError>).message });

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between py-3">
          <CollapsibleTrigger className="flex items-center gap-3 text-sm font-medium">
            {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            <IconTile icon={Network} tone="violet" />
            <span className="text-left">
              <CardTitle className="text-base">Correlation Analysis</CardTitle>
              <span className="block text-xs font-normal text-muted-foreground">How columns move together, plus mutual information and dependencies</span>
            </span>
          </CollapsibleTrigger>
          <Button size="sm" variant={triggered ? "outline" : "default"} onClick={handleRun}>
            {triggered ? "Re-run" : "Run Correlation"}
          </Button>
        </CardHeader>

        <CollapsibleContent>
          <CardContent className="pt-0 space-y-5">
            {!triggered && !isLoading && !data && !error && (
              <p className="text-sm text-muted-foreground">
                Click &quot;Run Correlation&quot; to detect high-correlation column pairs that may affect synthesis quality.
              </p>
            )}
            {isLoading && (
              <div className="space-y-2">
                <Skeleton className="h-64 w-full" />
                <Skeleton className="h-8 w-full" />
              </div>
            )}
            {error && <p className="text-sm text-destructive">Failed to compute correlation. Please try again.</p>}

            {data && (
              <>
                <div className="flex flex-wrap items-end gap-3">
                  <div className="space-y-1">
                    <Label htmlFor="corr-method">Method</Label>
                    <Select value={method} onValueChange={(v) => setMethod(v as CorrelationMethod)}>
                      <SelectTrigger id="corr-method" className="w-36">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="pearson">Pearson</SelectItem>
                        <SelectItem value="spearman">Spearman</SelectItem>
                        <SelectItem value="kendall">Kendall</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-1">
                    <Label htmlFor="corr-topk">Top pairs</Label>
                    <Select value={String(topK)} onValueChange={(v) => setTopK(Number(v))}>
                      <SelectTrigger id="corr-topk" className="w-24">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {[5, 10, 20].map((k) => (
                          <SelectItem key={k} value={String(k)}>
                            {k}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {method === "pearson" && data.heatmap_base64 ? (
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Pearson Correlation Heatmap</p>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={`data:image/png;base64,${data.heatmap_base64}`} alt="Correlation heatmap" className="w-full rounded border" />
                  </div>
                ) : (
                  <div>
                    <p className="text-xs text-muted-foreground mb-1 capitalize">{method} correlation matrix</p>
                    <MatrixTable matrix={data.matrices[method]} />
                  </div>
                )}

                <div>
                  <p className="text-sm font-medium mb-2">Top Correlated Pairs</p>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Column A</TableHead>
                        <TableHead>Column B</TableHead>
                        <TableHead>Correlation</TableHead>
                        <TableHead>Flag</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {data.top_pairs.map((pair, i) => (
                        <TableRow key={i}>
                          <TableCell className="font-mono text-xs">{pair.col_a}</TableCell>
                          <TableCell className="font-mono text-xs">{pair.col_b}</TableCell>
                          <TableCell className={`text-sm ${correlationColor(pair.correlation)}`}>{pair.correlation.toFixed(3)}</TableCell>
                          <TableCell>
                            {pair.is_leakage && (
                              <Badge variant="destructive" className="text-xs flex items-center gap-1 w-fit">
                                <AlertTriangle className="h-3 w-3" />
                                Data Leakage
                              </Badge>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>

                {data.categorical.length > 0 && (
                  <div>
                    <p className="text-sm font-medium mb-1">Categorical Associations (Cramér&apos;s V)</p>
                    <p className="text-xs text-muted-foreground mb-2">0 = independent, 1 = one column fully determines the other.</p>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Column A</TableHead>
                          <TableHead>Column B</TableHead>
                          <TableHead>Cramér&apos;s V</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {data.categorical.slice(0, topK).map((c, i) => (
                          <TableRow key={i}>
                            <TableCell className="font-mono text-xs">{c.col_a}</TableCell>
                            <TableCell className="font-mono text-xs">{c.col_b}</TableCell>
                            <TableCell className={`text-sm ${correlationColor(c.cramers_v)}`}>{c.cramers_v.toFixed(3)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                )}

                {/* Mutual information */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <div>
                      <p className="text-sm font-medium">Mutual information</p>
                      <p className="text-xs text-muted-foreground">Catches non-linear relationships that correlation misses.</p>
                    </div>
                    <Button size="sm" variant="outline" disabled={mi.isPending}
                      onClick={() => fileId && mi.mutate(fileId, { onSuccess: setMiResult, onError: failed("Mutual information") })}>
                      {mi.isPending && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1" />}
                      Compute
                    </Button>
                  </div>
                  {miResult && (
                    <>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Column A</TableHead>
                            <TableHead>Column B</TableHead>
                            <TableHead>MI (normalised)</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {miResult.top_mi_pairs.slice(0, topK).map((p, i) => (
                            <TableRow key={i}>
                              <TableCell className="font-mono text-xs">{p.column1}</TableCell>
                              <TableCell className="font-mono text-xs">{p.column2}</TableCell>
                              <TableCell className="text-sm">{p.mi_score.toFixed(3)}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                      {miResult.nonlinear_analysis.nonlinear_relationships.map((n, i) => (
                        <p key={i} className="text-xs text-amber-700">
                          Non-linear: <span className="font-mono">{n.column1}</span> ↔ <span className="font-mono">{n.column2}</span> (MI{" "}
                          {n.mi_score.toFixed(2)} vs Pearson {n.pearson_score.toFixed(2)})
                        </p>
                      ))}
                      {miResult.nonlinear_analysis.recommendations.map((r) => (
                        <p key={r} className="text-xs text-muted-foreground">{r}</p>
                      ))}
                    </>
                  )}
                </div>

                {/* Functional dependencies */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <div>
                      <p className="text-sm font-medium">Functional dependencies</p>
                      <p className="text-xs text-muted-foreground">Columns whose value fixes another&apos;s (e.g. country → country name).</p>
                    </div>
                    <Button size="sm" variant="outline" disabled={deps.isPending}
                      onClick={() => fileId && deps.mutate(fileId, { onSuccess: setDepResult, onError: failed("Dependency detection") })}>
                      {deps.isPending && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1" />}
                      Detect
                    </Button>
                  </div>
                  {depResult && (
                    <>
                      <p className="text-xs text-muted-foreground">
                        {depResult.graph_data.summary.total_dependencies} dependencies across{" "}
                        {depResult.graph_data.summary.columns_involved} columns ({depResult.graph_data.summary.strict_dependencies} one-to-one)
                      </p>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Determines</TableHead>
                            <TableHead>Dependent</TableHead>
                            <TableHead>Type</TableHead>
                            <TableHead>Confidence</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {depResult.dependencies.slice(0, 15).map((d, i) => (
                            <TableRow key={i}>
                              <TableCell className="font-mono text-xs">{d.determinant}</TableCell>
                              <TableCell className="font-mono text-xs">{d.dependent}</TableCell>
                              <TableCell className="text-xs">{d.type}</TableCell>
                              <TableCell className="text-xs">{(d.confidence * 100).toFixed(0)}%</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
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
