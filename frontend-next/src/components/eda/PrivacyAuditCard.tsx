"use client";

import { useState } from "react";
import { usePrivacy } from "@/hooks/api/usePrivacy";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ChevronDown, ChevronRight, Shield, ShieldAlert, ShieldCheck } from "lucide-react";
import { IconTile } from "@/components/ui/icon-tile";
import { RadialBarChart, RadialBar, PolarAngleAxis, ResponsiveContainer, Tooltip } from "recharts";
import type { AppError, KAnonymityResponse, QuasiIdentifier } from "@/lib/api/types";
import { useKAnonymity, useQuasiIdentifiers } from "@/hooks/api/useAnalysis";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";

function KAnonymityGauge({ k, threshold }: { k: number; threshold: number }) {
  const pct = Math.min(100, (k / Math.max(threshold * 2, 1)) * 100);
  const color = k >= threshold ? "#22c55e" : k >= threshold * 0.5 ? "#f59e0b" : "#ef4444";
  const data = [{ name: "k", value: pct, fill: color }];
  return (
    <div className="flex flex-col items-center">
      <div className="w-32 h-32">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            cx="50%"
            cy="50%"
            innerRadius="60%"
            outerRadius="80%"
            data={data}
            startAngle={90}
            endAngle={-270}
          >
            <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
            <RadialBar dataKey="value" background />
            <Tooltip formatter={(v) => `${Number(v).toFixed(0)}%`} />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-2xl font-bold" style={{ color }}>
        k = {k}
      </p>
      <p className="text-xs text-muted-foreground">threshold: {threshold}</p>
    </div>
  );
}

/** Re-identification risk: pick quasi-identifiers, then measure k-anonymity over them */
function KAnonymitySection({ fileId }: { fileId: string }) {
  const qis = useQuasiIdentifiers();
  const kanon = useKAnonymity();
  const [candidates, setCandidates] = useState<QuasiIdentifier[] | null>(null);
  const [picked, setPicked] = useState<string[]>([]);
  const [k, setK] = useState(5);
  const [result, setResult] = useState<KAnonymityResponse | null>(null);
  const failed = (what: string) => (e: unknown) => toast.error(`${what} failed`, { description: (e as Partial<AppError>).message });

  function suggest() {
    qis.mutate(fileId, {
      onSuccess: (list) => {
        setCandidates(list);
        const risky = list.filter((q) => q.risk_level !== "Low").map((q) => q.column);
        setPicked(risky.length ? risky : list.map((q) => q.column));
        setResult(null);
      },
      onError: failed("Quasi-identifier detection"),
    });
  }

  return (
    <div className="space-y-3 border-t pt-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-sm font-medium">Re-identification risk (k-anonymity)</p>
          <p className="text-xs text-muted-foreground">
            Quasi-identifiers are columns that identify people in combination. A record is k-anonymous if at
            least k−1 others share its values.
          </p>
        </div>
        <Button size="sm" variant="outline" onClick={suggest} disabled={qis.isPending}>
          {qis.isPending && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1" />}
          Suggest quasi-identifiers
        </Button>
      </div>

      {candidates && candidates.length === 0 && (
        <p className="text-sm text-muted-foreground">No quasi-identifier candidates found.</p>
      )}
      {candidates && candidates.length > 0 && (
        <div className="flex flex-wrap items-end gap-4">
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            {candidates.map((q) => (
              <div key={q.column} className="flex items-center gap-1.5">
                <Checkbox
                  id={`qi-${q.column}`}
                  checked={picked.includes(q.column)}
                  onCheckedChange={() =>
                    setPicked((p) => (p.includes(q.column) ? p.filter((c) => c !== q.column) : [...p, q.column]))
                  }
                />
                <Label htmlFor={`qi-${q.column}`} className="text-xs cursor-pointer">
                  <span className="font-mono">{q.column}</span>{" "}
                  <span className="text-muted-foreground">
                    ({q.unique_values} values, {q.risk_level} risk)
                  </span>
                </Label>
              </div>
            ))}
          </div>
          <div className="flex items-end gap-2">
            <div className="space-y-1">
              <Label htmlFor="k-threshold">k threshold</Label>
              <Input id="k-threshold" type="number" min={2} className="h-8 w-20" value={k} onChange={(e) => setK(Math.max(2, Number(e.target.value) || 2))} />
            </div>
            <Button size="sm" disabled={picked.length === 0 || kanon.isPending}
              onClick={() => kanon.mutate({ fileId, quasiIdentifiers: picked, k }, { onSuccess: setResult, onError: failed("k-anonymity") })}>
              {kanon.isPending && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1" />}
              Compute k-anonymity
            </Button>
          </div>
        </div>
      )}

      {result && (
        <div className="flex flex-col md:flex-row gap-6 items-start">
          <KAnonymityGauge k={result.risk_report.statistics.min_k} threshold={result.k_threshold} />
          <div className="space-y-1 text-sm">
            <p>
              Severity: <span className="font-medium">{result.risk_report.summary.severity}</span>
            </p>
            <p>
              <span className="font-medium">{result.vulnerable_count.toLocaleString()}</span> of{" "}
              {result.total_records.toLocaleString()} records have fewer than {result.k_threshold} look-alikes.
            </p>
            <p className="text-muted-foreground text-xs">
              Unique records: {result.risk_report.summary.unique_records} ({result.risk_report.statistics.unique_percentage}%) · average k:{" "}
              {result.risk_report.summary.average_k}
            </p>
            {result.risk_report.recommendations.map((r) => (
              <p key={r.issue} className="text-xs">
                <span className="font-medium">{r.issue}:</span> {r.recommendation}
              </p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export function PrivacyAuditCard() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const [open, setOpen] = useState(false);
  const [triggered, setTriggered] = useState(false);
  const [mode, setMode] = useState<"fast" | "deep">("fast");

  const { data, isLoading, error, refetch } = usePrivacy(
    triggered ? fileId : null,
    mode
  );

  function handleRun() {
    setTriggered(true);
    setOpen(true);
    if (triggered) refetch();
  }

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between py-3">
          <CollapsibleTrigger
              className="flex items-center gap-3 text-sm font-medium"
            >
              {open ? (
                  <ChevronDown className="h-4 w-4" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
              <IconTile icon={ShieldCheck} tone="emerald" />
              <span className="text-left">
                <CardTitle className="text-base">Privacy Audit</CardTitle>
                <span className="block text-xs font-normal text-muted-foreground">Personal data, quasi-identifiers and k-anonymity</span>
              </span>
            </CollapsibleTrigger>
          <div className="flex items-center gap-2">
            <div className="flex rounded-lg bg-muted p-0.5 text-xs font-medium ring-1 ring-inset ring-border/60">
              <button
                className={`rounded-md px-2.5 py-1 transition-colors ${mode === "fast" ? "bg-background text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
                onClick={() => setMode("fast")}
                aria-pressed={mode === "fast"}
              >
                Fast
              </button>
              <button
                className={`rounded-md px-2.5 py-1 transition-colors ${mode === "deep" ? "bg-background text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
                onClick={() => setMode("deep")}
                aria-pressed={mode === "deep"}
              >
                Deep
              </button>
            </div>
            <Button size="sm" variant={triggered ? "outline" : "default"} onClick={handleRun}>
              {triggered ? "Re-scan" : "Run Scan"}
            </Button>
          </div>
        </CardHeader>

        <CollapsibleContent>
          <CardContent className="pt-0 space-y-4">
            {mode === "deep" && !triggered && (
              <p className="text-xs text-amber-600 flex items-center gap-1">
                <ShieldAlert className="h-3.5 w-3.5" />
                Deep scan uses Presidio AI and may take 30 seconds to 5 minutes.
              </p>
            )}
            {isLoading && (
              <div className="space-y-2">
                <Skeleton className="h-32 w-32 mx-auto rounded-full" />
                <Skeleton className="h-8 w-full" />
                <Skeleton className="h-8 w-full" />
              </div>
            )}
            {error && (
              <p className="text-sm text-destructive">
                Privacy scan failed: {(error as Partial<AppError>).message ?? "please try again."}
              </p>
            )}
            {data && (
              <div className="space-y-4">
                {data.notice && (
                  <p className="text-xs text-amber-600 flex items-start gap-1">
                    <ShieldAlert className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                    {data.notice}
                  </p>
                )}
                <div className="flex flex-col md:flex-row gap-6 items-start">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <Shield className="h-4 w-4 text-blue-500" />
                      <p className="text-sm font-medium">PII Detection</p>
                    </div>
                    {data.pii_results.length === 0 ? (
                      <p className="text-sm text-green-600">No PII detected.</p>
                    ) : (
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Column</TableHead>
                            <TableHead>PII Type</TableHead>
                            <TableHead>Confidence</TableHead>
                            <TableHead>Sample (masked)</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {data.pii_results.map((r, i) => (
                            <TableRow key={i}>
                              <TableCell className="font-mono text-xs">{r.column}</TableCell>
                              <TableCell>
                                <Badge variant="destructive" className="text-xs">
                                  {r.pii_type}
                                </Badge>
                              </TableCell>
                              <TableCell className="text-xs">
                                {(r.confidence * 100).toFixed(0)}%
                              </TableCell>
                              <TableCell className="font-mono text-xs text-muted-foreground">
                                {r.sample_masked}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    )}
                  </div>

                </div>
              </div>
            )}
            {fileId && <KAnonymitySection fileId={fileId} />}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
