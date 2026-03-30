"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
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
import { ChevronDown, ChevronRight, TrendingUp } from "lucide-react";

interface TimeSeriesDetectionResult {
  detected_columns?: string[];
  time_column?: string;
  frequency?: string;
  is_regular?: boolean;
  metadata?: Record<string, unknown>;
}

function useTimeSeriesDetection(fileId: string | null, enabled: boolean) {
  return useQuery<TimeSeriesDetectionResult>({
    queryKey: ["timeseries-detection", fileId],
    queryFn: () =>
      apiClient
        .post<TimeSeriesDetectionResult>(`/eda/detect-timeseries/${fileId}`)
        .then((r) => r.data),
    enabled: !!fileId && enabled,
    staleTime: 10 * 60 * 1000,
    retry: 1,
  });
}

export function TimeSeriesCard() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const [open, setOpen] = useState(false);
  const [triggered, setTriggered] = useState(false);

  const { data, isLoading, error, refetch } = useTimeSeriesDetection(
    triggered ? fileId : null,
    triggered
  );

  function handleRun() {
    setTriggered(true);
    setOpen(true);
    if (triggered) refetch();
  }

  const detectedCols = data?.detected_columns ?? [];

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between py-3">
          <CollapsibleTrigger className="flex items-center gap-2 text-sm font-medium">
            {open ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
            <TrendingUp className="h-4 w-4 text-blue-500" />
            <CardTitle className="text-base">Time Series Detection</CardTitle>
          </CollapsibleTrigger>
          <Button size="sm" variant="outline" onClick={handleRun}>
            {triggered ? "Refresh" : "Detect"}
          </Button>
        </CardHeader>

        <CollapsibleContent>
          <CardContent className="pt-0 space-y-4">
            {isLoading && (
              <div className="space-y-2">
                {[...Array(2)].map((_, i) => (
                  <Skeleton key={i} className="h-10 w-full" />
                ))}
              </div>
            )}
            {error && (
              <p className="text-sm text-destructive">
                Time series detection failed. The dataset may not contain temporal data.
              </p>
            )}
            {data && (
              <div className="space-y-3">
                {detectedCols.length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    No time series columns detected in this dataset.
                  </p>
                ) : (
                  <>
                    <div className="flex flex-wrap gap-2">
                      {detectedCols.map((col) => (
                        <Badge key={col} variant="secondary" className="font-mono text-xs">
                          {col}
                        </Badge>
                      ))}
                    </div>
                    {data.time_column && (
                      <div className="rounded border px-3 py-2 text-sm space-y-1">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Time column</span>
                          <span className="font-mono font-medium">{data.time_column}</span>
                        </div>
                        {data.frequency && (
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Frequency</span>
                            <span className="font-medium">{data.frequency}</span>
                          </div>
                        )}
                        {data.is_regular !== undefined && (
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Regular intervals</span>
                            <Badge
                              variant={data.is_regular ? "default" : "outline"}
                              className="text-xs"
                            >
                              {data.is_regular ? "Yes" : "No"}
                            </Badge>
                          </div>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
