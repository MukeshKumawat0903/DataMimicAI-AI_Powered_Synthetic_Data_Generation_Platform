"use client";

import { useState, useMemo } from "react";
import { useFeatures } from "@/hooks/api/useFeatures";
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
import { ChevronDown, ChevronRight, Lightbulb, AlertOctagon } from "lucide-react";

const IMPACT_COLOR: Record<string, string> = {
  high: "bg-red-100 text-red-700 border-red-200",
  medium: "bg-amber-100 text-amber-700 border-amber-200",
  low: "bg-blue-100 text-blue-700 border-blue-200",
};

export function FeatureSuggestionsCard() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const [open, setOpen] = useState(false);
  const [triggered, setTriggered] = useState(false);

  const { data, isLoading, error, refetch } = useFeatures(triggered ? fileId : null);

  // P2.2: memoize combined suggestion list
  const allSuggestions = useMemo(
    () => [
      ...(data?.utility_suggestions ?? []),
      ...(data?.privacy_suggestions ?? []),
    ],
    [data]
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
              <CardTitle className="text-base">Feature Suggestions</CardTitle>
            </CollapsibleTrigger>
          <Button size="sm" variant="outline" onClick={handleRun}>
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
            {error && (
              <p className="text-sm text-destructive">Failed to analyse features. Please try again.</p>
            )}
            {/* P2.2: use memoized allSuggestions from hook above */}
            {data && (
              <div className="space-y-4">
                {/* Utility Suggestions */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <Lightbulb className="h-4 w-4 text-amber-500" />
                    <p className="text-sm font-medium">
                      Utility Suggestions ({(data.utility_suggestions ?? []).length})
                    </p>
                  </div>
                  {(data.utility_suggestions ?? []).length === 0 ? (
                    <p className="text-sm text-muted-foreground">No utility suggestions.</p>
                  ) : (
                    <div className="space-y-2">
                      {(data.utility_suggestions ?? []).map((s, i) => (
                        <div
                          key={i}
                          className={`flex items-start justify-between rounded border px-3 py-2 text-sm ${IMPACT_COLOR[s.impact]}`}
                        >
                          <div>
                            <span className="font-mono font-medium">{s.column}</span>
                            <span className="mx-1 text-muted-foreground">—</span>
                            <span>{s.description}</span>
                          </div>
                          <Badge variant="outline" className="text-xs shrink-0 ml-2">
                            {s.suggestion_type}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Privacy Suggestions */}
                {(data.privacy_suggestions ?? []).length > 0 && (
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <Lightbulb className="h-4 w-4 text-blue-500" />
                      <p className="text-sm font-medium">
                        Privacy Suggestions ({(data.privacy_suggestions ?? []).length})
                      </p>
                    </div>
                    <div className="space-y-2">
                      {(data.privacy_suggestions ?? []).map((s, i) => (
                        <div
                          key={i}
                          className={`flex items-start justify-between rounded border px-3 py-2 text-sm ${IMPACT_COLOR[s.impact]}`}
                        >
                          <div>
                            <span className="font-mono font-medium">{s.column}</span>
                            <span className="mx-1 text-muted-foreground">—</span>
                            <span>{s.description}</span>
                          </div>
                          <Badge variant="outline" className="text-xs shrink-0 ml-2">
                            {s.suggestion_type}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Conflicts */}
                {(data.conflicts ?? []).length > 0 && (
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <AlertOctagon className="h-4 w-4 text-destructive" />
                      <p className="text-sm font-medium">
                        Conflicts ({(data.conflicts ?? []).length})
                      </p>
                    </div>
                    <div className="space-y-2">
                      {(data.conflicts ?? []).map((c, i) => (
                        <div
                          key={i}
                          className="rounded border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive"
                        >
                          <span className="font-mono font-medium">
                            {c.column ?? c.col_a}
                          </span>
                          {c.col_b && (
                            <>
                              {" ↔ "}
                              <span className="font-mono font-medium">{c.col_b}</span>
                            </>
                          )}
                          {": "}
                          {c.reason ?? c.privacy ?? c.utility}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {allSuggestions.length === 0 && (data.conflicts ?? []).length === 0 && (
                  <p className="text-sm text-muted-foreground">No feature suggestions found.</p>
                )}
              </div>
            )}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
