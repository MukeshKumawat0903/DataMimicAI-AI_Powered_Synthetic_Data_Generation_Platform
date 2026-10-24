"use client";

import { useState } from "react";
import { useLLMExplanation, useLLMScopes } from "@/hooks/api/useLLMExplanation";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AlertTriangle, Bot, CheckCircle2, Loader2, RefreshCw, XCircle } from "lucide-react";
import { IconTile } from "@/components/ui/icon-tile";
import { humanize } from "@/lib/utils";
import type { AppError, LLMExplanationOptions } from "@/lib/api/types";

export function LLMExplanationPanel() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const columns = useWorkspaceStore((s) => s.columns);
  const { data: scopes } = useLLMScopes();

  const [scope, setScope] = useState("dataset_overview");
  const [tone, setTone] = useState("clear");
  const [picked, setPicked] = useState<string[]>([]);
  // The options of the last run; the query only fires once the user asks for it
  const [request, setRequest] = useState<LLMExplanationOptions | null>(null);

  const needsColumns = scopes?.scopes[scope]?.requires_columns ?? false;
  const { data, isFetching, error, refetch } = useLLMExplanation(
    fileId,
    request ?? { scope, tone },
    request !== null
  );

  function handleRun() {
    const next = { scope, tone, columns: needsColumns ? picked : undefined };
    if (JSON.stringify(next) === JSON.stringify(request)) refetch();
    else setRequest(next);
  }

  function togglePicked(col: string) {
    setPicked((prev) => (prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]));
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-start gap-3 py-3">
        <IconTile icon={Bot} />
        <div className="space-y-1">
          <CardTitle className="text-base">AI Diagnostics</CardTitle>
          <p className="text-sm text-muted-foreground">
            An LLM explains what the analysis found, grounded in the computed statistics. Every answer is
            checked before it&apos;s shown.
          </p>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="llm-scope">Focus</Label>
            <Select value={scope} onValueChange={(v) => setScope(v as string)}>
              <SelectTrigger id="llm-scope" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Object.keys(scopes?.scopes ?? { dataset_overview: null }).map((key) => (
                  <SelectItem key={key} value={key}>
                    {humanize(key)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {scopes?.scopes[scope] && (
              <p className="text-xs text-muted-foreground">{scopes.scopes[scope].description}</p>
            )}
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="llm-tone">Tone</Label>
            <Select value={tone} onValueChange={(v) => setTone(v as string)}>
              <SelectTrigger id="llm-tone" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Object.keys(scopes?.tones ?? { clear: "" }).map((key) => (
                  <SelectItem key={key} value={key}>
                    {humanize(key)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {scopes?.tones[tone] && <p className="text-xs text-muted-foreground">{scopes.tones[tone]}</p>}
          </div>
        </div>

        {needsColumns && (
          <div className="space-y-1.5">
            <Label>Columns to analyse</Label>
            <div className="flex flex-wrap gap-x-4 gap-y-2">
              {columns.map((col) => (
                <div key={col} className="flex items-center gap-1.5">
                  <Checkbox
                    id={`llm-col-${col}`}
                    checked={picked.includes(col)}
                    onCheckedChange={() => togglePicked(col)}
                  />
                  <Label htmlFor={`llm-col-${col}`} className="text-sm font-mono cursor-pointer">
                    {col}
                  </Label>
                </div>
              ))}
            </div>
          </div>
        )}

        <Button
          onClick={handleRun}
          disabled={isFetching || (needsColumns && picked.length === 0)}
          className="gap-1.5"
        >
          {isFetching ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : request ? (
            <RefreshCw className="h-3.5 w-3.5" />
          ) : null}
          {isFetching ? "Analysing…" : request ? "Run again" : "Run AI analysis"}
        </Button>

        {isFetching && !data && (
          <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-5/6" />
            <Skeleton className="h-4 w-4/6" />
          </div>
        )}

        {error && (
          <p className="text-sm text-destructive flex items-center gap-1">
            <AlertTriangle className="h-3.5 w-3.5" />
            AI analysis failed: {(error as Partial<AppError>).message ?? "please try again."}
          </p>
        )}

        {data && (
          <div className="space-y-3">
            {!data.validated && data.source === "facts_summary" && (
              <div className="flex items-start gap-2 rounded border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-700">
                <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                The AI&apos;s answer didn&apos;t pass the checks below, so it isn&apos;t shown. Instead, here
                is a summary built directly from the computed statistics.
              </div>
            )}
            {!data.validated && data.source === "llm" && (
              <div className="flex items-start gap-2 rounded border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-700">
                <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                The answer didn&apos;t pass all checks ({data.validation_pct}%), so it may be generic or
                inaccurate. Review it against the analysis results.
              </div>
            )}
            <div className="flex flex-wrap gap-3">
              {data.checks.map((c) => (
                <span key={c.label} className="flex items-center gap-1 text-xs">
                  {c.passed ? (
                    <CheckCircle2 className="h-3.5 w-3.5 text-green-600" />
                  ) : (
                    <XCircle className="h-3.5 w-3.5 text-destructive" />
                  )}
                  {c.label}
                </span>
              ))}
              <Badge variant="outline" className="text-xs ml-auto">
                {data.source === "facts_summary" ? "Summary from statistics" : `${data.validation_pct}% of checks passed`}
              </Badge>
            </div>
            <div className="rounded-xl border bg-gradient-to-br from-violet-50/70 to-card p-4 dark:from-violet-500/10 dark:to-card">
              <p className="text-sm leading-relaxed whitespace-pre-wrap">{data.explanation}</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
