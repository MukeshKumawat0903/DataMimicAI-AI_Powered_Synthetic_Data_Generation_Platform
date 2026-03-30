"use client";

import { useState } from "react";
import { useLLMExplanation } from "@/hooks/api/useLLMExplanation";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertTriangle, Bot, CheckCircle2, Circle, Loader2, RefreshCw } from "lucide-react";

const STEP_LABELS = [
  "Loading dataset",
  "Detecting issues",
  "Gathering context",
  "Querying LLM",
  "Parsing response",
  "Validating facts",
  "Preparing output",
];

export function LLMExplanationPanel() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const [enabled, setEnabled] = useState(false);

  const { data, isLoading, error, refetch } = useLLMExplanation(fileId, enabled);

  function handleRun() {
    if (enabled) {
      refetch();
    } else {
      setEnabled(true);
    }
  }

  const steps = data?.steps ?? STEP_LABELS.map((label, i) => ({
    step: i + 1,
    label,
    status: "pending" as const,
  }));

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between py-3">
        <div className="flex items-center gap-2">
          <Bot className="h-4 w-4 text-blue-500" />
          <CardTitle className="text-base">AI Diagnostics</CardTitle>
        </div>
        <Button
          size="sm"
          variant={enabled ? "outline" : "default"}
          onClick={handleRun}
          disabled={isLoading}
        >
          {isLoading ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin mr-1" />
          ) : enabled ? (
            <RefreshCw className="h-3.5 w-3.5 mr-1" />
          ) : null}
          {isLoading ? "Analysing…" : enabled ? "Regenerate" : "Run AI Analysis"}
        </Button>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Pipeline steps */}
        <div className="flex gap-2 flex-wrap">
          {steps.map((s) => (
            <div
              key={s.step}
              className="flex items-center gap-1 text-xs px-2 py-1 rounded-full border"
            >
              {s.status === "done" ? (
                <CheckCircle2 className="h-3 w-3 text-green-500" />
              ) : s.status === "running" ? (
                <Loader2 className="h-3 w-3 animate-spin text-blue-500" />
              ) : s.status === "error" ? (
                <AlertTriangle className="h-3 w-3 text-destructive" />
              ) : (
                <Circle className="h-3 w-3 text-muted-foreground" />
              )}
              <span
                className={
                  s.status === "done"
                    ? "text-green-700"
                    : s.status === "running"
                    ? "text-blue-700"
                    : "text-muted-foreground"
                }
              >
                {s.label}
              </span>
            </div>
          ))}
        </div>

        {isLoading && !data && (
          <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-5/6" />
            <Skeleton className="h-4 w-4/6" />
          </div>
        )}

        {error && (
          <p className="text-sm text-destructive flex items-center gap-1">
            <AlertTriangle className="h-3.5 w-3.5" />
            AI analysis failed. Check backend logs and try again.
          </p>
        )}

        {data && (
          <div className="space-y-3">
            {data.validation_pct < 70 && (
              <div className="flex items-center gap-2 rounded border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-700">
                <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                Low validation confidence ({data.validation_pct.toFixed(0)}%). Some facts may be
                inaccurate — review manually.
              </div>
            )}
            <div className="flex justify-between items-center">
              <p className="text-sm font-medium">Explanation</p>
              <Badge variant="outline" className="text-xs">
                {data.validation_pct.toFixed(0)}% validated
              </Badge>
            </div>
            <p className="text-sm leading-relaxed whitespace-pre-wrap text-muted-foreground">
              {data.explanation}
            </p>
          </div>
        )}

        {!enabled && !isLoading && (
          <p className="text-sm text-muted-foreground">
            Click &ldquo;Run AI Analysis&rdquo; to generate an AI-powered explanation of your dataset&apos;s quality issues.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
