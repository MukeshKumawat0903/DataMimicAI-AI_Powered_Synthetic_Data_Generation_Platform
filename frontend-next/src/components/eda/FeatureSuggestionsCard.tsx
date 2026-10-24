"use client";

import { useState } from "react";
import { useFeatures } from "@/hooks/api/useFeatures";
import { useResolveConflict } from "@/hooks/api/useTransforms";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { TransformActions } from "./TransformActions";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { AlertOctagon, Check, ChevronDown, ChevronRight, Lightbulb, Lock, Sparkles } from "lucide-react";
import { IconTile } from "@/components/ui/icon-tile";
import { toast } from "sonner";
import { cn, humanize } from "@/lib/utils";
import type { FeatureConflict, FeatureSuggestion, TransformDecision } from "@/lib/api/types";

const IMPACT_COLOR: Record<string, string> = {
  high: "bg-red-100 text-red-700 border-red-200",
  medium: "bg-amber-100 text-amber-700 border-amber-200",
  low: "bg-blue-100 text-blue-700 border-blue-200",
};

const toDecision = (s: FeatureSuggestion): TransformDecision => ({
  column: s.column,
  transformation: s.suggestion_type,
  category: s.category,
  params: s.params,
  reason: s.description,
  metadata: s.metrics,
});

function SuggestionRow({ s, on, onToggle }: { s: FeatureSuggestion; on: boolean; onToggle: () => void }) {
  return (
    <div className={cn("flex items-start justify-between gap-2 rounded border px-3 py-2 text-sm", IMPACT_COLOR[s.impact])}>
      <div>
        <span className="font-mono font-medium">{s.column}</span>
        <span className="mx-1 text-muted-foreground">—</span>
        <span>{s.description}</span>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <Badge variant="outline" className="text-xs">
          {humanize(s.suggestion_type)}
        </Badge>
        <Button size="sm" variant={on ? "default" : "outline"} className="h-7 gap-1" aria-pressed={on} onClick={onToggle}>
          {on && <Check className="h-3.5 w-3.5" />}
          {on ? "Accepted" : "Accept"}
        </Button>
      </div>
    </div>
  );
}

function Tile({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg border px-3 py-2">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-xl font-semibold tabular-nums">{value}</p>
    </div>
  );
}

export function FeatureSuggestionsCard() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const accepted = useWorkspaceStore((s) => s.acceptedDecisions);
  const { toggleDecision, setDecisions } = useWorkspaceStore();
  const [open, setOpen] = useState(false);
  const [triggered, setTriggered] = useState(false);
  const { data, isLoading, error, refetch } = useFeatures(triggered ? fileId : null);
  const resolve = useResolveConflict();

  const isAccepted = (s: FeatureSuggestion) =>
    accepted.some((d) => d.column === s.column && d.category === s.category && d.transformation === s.suggestion_type);

  function handleRun() {
    setTriggered(true);
    setOpen(true);
    if (triggered) refetch();
  }

  function keep(conflict: FeatureConflict, choice: FeatureSuggestion) {
    if (!fileId) return;
    resolve.mutate(
      { fileId, column: conflict.column, category: choice.category, transformation: choice.suggestion_type },
      {
        onSuccess: () => {
          // Keep only the chosen side for this column
          const others = accepted.filter((d) => d.column !== conflict.column);
          setDecisions([...others, toDecision(choice)]);
        },
        onError: () => toast.error("Couldn't record the choice"),
      }
    );
  }

  const row = (s: FeatureSuggestion, i: number) => (
    <SuggestionRow key={i} s={s} on={isAccepted(s)} onToggle={() => toggleDecision(toDecision(s))} />
  );

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between py-3">
          <CollapsibleTrigger className="flex items-center gap-3 text-sm font-medium">
            {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            <IconTile icon={Sparkles} tone="pink" />
            <span className="text-left">
              <CardTitle className="text-base">Feature Suggestions</CardTitle>
              <span className="block text-xs font-normal text-muted-foreground">Utility and privacy transforms you can accept and apply</span>
            </span>
          </CollapsibleTrigger>
          <Button size="sm" variant={triggered ? "outline" : "default"} onClick={handleRun}>
            {triggered ? "Refresh" : "Analyse Features"}
          </Button>
        </CardHeader>

        <CollapsibleContent>
          <CardContent className="pt-0 space-y-4">
            {isLoading && (
              <div className="space-y-2">
                {[...Array(3)].map((_, i) => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            )}
            {error && <p className="text-sm text-destructive">Failed to analyse features. Please try again.</p>}

            {data && (
              <div className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  Utility suggestions improve data quality for modelling; privacy suggestions reduce
                  re-identification risk. Accept the ones you want, then preview their impact or apply them to
                  generation.
                </p>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  <Tile label="Utility suggestions" value={data.utility_suggestions.length} />
                  <Tile label="Privacy suggestions" value={data.privacy_suggestions.length} />
                  <Tile label="Conflicts" value={data.conflicts.length} />
                  <Tile label="Decisions made" value={accepted.length} />
                </div>

                <div className="space-y-2">
                  <p className="text-sm font-medium flex items-center gap-2">
                    <Lightbulb className="h-4 w-4 text-amber-500" />
                    Utility Suggestions ({data.utility_suggestions.length})
                  </p>
                  {data.utility_suggestions.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No utility suggestions.</p>
                  ) : (
                    data.utility_suggestions.map(row)
                  )}
                </div>

                {data.privacy_suggestions.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-sm font-medium flex items-center gap-2">
                      <Lock className="h-4 w-4 text-blue-500" />
                      Privacy Suggestions ({data.privacy_suggestions.length})
                    </p>
                    {data.privacy_suggestions.map(row)}
                  </div>
                )}

                {data.conflicts.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-sm font-medium flex items-center gap-2">
                      <AlertOctagon className="h-4 w-4 text-destructive" />
                      Conflicts ({data.conflicts.length})
                    </p>
                    {data.conflicts.map((c, i) => {
                      const chosen = accepted.find((d) => d.column === c.column);
                      return (
                        <div key={i} className="rounded border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm space-y-2">
                          <p>
                            <span className="font-mono font-medium">{c.column}</span>: {c.reason}
                          </p>
                          <div className="flex flex-wrap items-center gap-2">
                            {[c.utility, c.privacy].filter(Boolean).map((s) => (
                              <Button
                                key={s!.category}
                                size="sm"
                                variant={chosen?.category === s!.category ? "default" : "outline"}
                                className="h-7"
                                disabled={resolve.isPending}
                                onClick={() => keep(c, s!)}
                              >
                                Keep {s!.category}: {humanize(s!.suggestion_type)}
                              </Button>
                            ))}
                            {chosen && <span className="text-xs text-muted-foreground">Resolved: {chosen.category}</span>}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}

                {accepted.length > 0 && <TransformActions />}
              </div>
            )}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
