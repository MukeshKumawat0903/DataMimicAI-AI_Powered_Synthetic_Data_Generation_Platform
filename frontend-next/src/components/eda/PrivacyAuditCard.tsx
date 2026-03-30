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
import { ChevronDown, ChevronRight, Shield, ShieldAlert } from "lucide-react";
import { RadialBarChart, RadialBar, ResponsiveContainer, Tooltip } from "recharts";

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
              className="flex items-center gap-2 text-sm font-medium"
            >
              {open ? (
                  <ChevronDown className="h-4 w-4" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
              <CardTitle className="text-base">Privacy Audit</CardTitle>
            </CollapsibleTrigger>
          <div className="flex items-center gap-2">
            <div className="flex text-xs border rounded overflow-hidden">
              <button
                className={`px-2 py-1 ${mode === "fast" ? "bg-primary text-primary-foreground" : ""}`}
                onClick={() => setMode("fast")}
                aria-pressed={mode === "fast"}
              >
                Fast
              </button>
              <button
                className={`px-2 py-1 ${mode === "deep" ? "bg-primary text-primary-foreground" : ""}`}
                onClick={() => setMode("deep")}
                aria-pressed={mode === "deep"}
              >
                Deep
              </button>
            </div>
            <Button size="sm" variant="outline" onClick={handleRun}>
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
                Privacy scan failed. Please try again.
              </p>
            )}
            {data && (
              <div className="space-y-4">
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

                  <div className="flex-shrink-0">
                    <p className="text-sm font-medium mb-2 text-center">k-Anonymity</p>
                    <KAnonymityGauge
                      k={data.k_anonymity.k_value}
                      threshold={data.k_anonymity.threshold}
                    />
                    <p className="text-xs text-muted-foreground mt-1 text-center">
                      {data.k_anonymity.at_risk_records} records at risk
                    </p>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
