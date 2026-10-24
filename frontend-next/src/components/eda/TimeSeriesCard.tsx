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
import { IconTile } from "@/components/ui/icon-tile";
import { TimeSeriesAnalysis } from "./TimeSeriesAnalysis";

interface TimeSeriesDetectionResult {
  detected_columns?: string[];
  time_column?: string;
  frequency?: string;
  is_regular?: boolean;
  /** Numeric columns worth analysing over time */
  targets?: string[];
  /** Distinct timestamps, and how many rows share each one (> 1 when the file holds several series) */
  periods?: number;
  rows_per_timestamp?: number;
  metadata?: Record<string, unknown>;
}

const FREQUENCY_LABELS: Record<string, string> = {
  D: "Daily",
  B: "Business days",
  W: "Weekly",
  M: "Monthly",
  MS: "Monthly",
  ME: "Monthly",
  Q: "Quarterly",
  QS: "Quarterly",
  Y: "Yearly",
  A: "Yearly",
  H: "Hourly",
  h: "Hourly",
  T: "Every minute",
  min: "Every minute",
  S: "Every second",
  s: "Every second",
};

/** pandas frequency alias ("W-SUN", "QS-OCT", "15T") → readable label */
function frequencyLabel(freq: string) {
  const base = freq.split("-")[0];
  const step = base.match(/^\d+/)?.[0];
  const label = FREQUENCY_LABELS[base.replace(/^\d+/, "")];
  if (!label) return freq;
  return step && step !== "1" ? `${label} ×${step}` : label;
}

// Backend response shape from POST /eda/detect-timeseries/{file_id}
interface BackendTimeSeriesResponse {
  detection?: {
    datetime_columns?: string[];
    primary_datetime_column?: string | null;
    frequency?: string | null;
    is_regular?: boolean;
    potential_targets?: string[];
    periods?: number;
    rows_per_timestamp?: number;
  };
}

function useTimeSeriesDetection(fileId: string | null, enabled: boolean) {
  return useQuery<TimeSeriesDetectionResult>({
    queryKey: ["timeseries-detection", fileId],
    queryFn: () =>
      apiClient
        .post<BackendTimeSeriesResponse>(`/eda/detect-timeseries/${fileId}`)
        .then(({ data: { detection = {} } }) => ({
          detected_columns: detection.datetime_columns ?? [],
          time_column: detection.primary_datetime_column ?? undefined,
          frequency: detection.frequency ?? undefined,
          is_regular: detection.is_regular,
          targets: detection.potential_targets ?? [],
          periods: detection.periods,
          rows_per_timestamp: detection.rows_per_timestamp,
        })),
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
          <CollapsibleTrigger className="flex items-center gap-3 text-sm font-medium">
            {open ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
            <IconTile icon={TrendingUp} tone="cyan" />
            <span className="text-left">
              <CardTitle className="text-base">Time Series Detection</CardTitle>
              <span className="block text-xs font-normal text-muted-foreground">Date columns, autocorrelation, decomposition and outliers</span>
            </span>
          </CollapsibleTrigger>
          <Button size="sm" variant={triggered ? "outline" : "default"} onClick={handleRun}>
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
                            <span className="font-medium">{frequencyLabel(data.frequency)}</span>
                          </div>
                        )}
                        {data.periods !== undefined && (
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Timestamps</span>
                            <span className="font-medium">{data.periods.toLocaleString()}</span>
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
                    {(data.rows_per_timestamp ?? 1) > 1 && (
                      <p className="text-xs text-muted-foreground">
                        Each timestamp has about {Math.round(data.rows_per_timestamp ?? 1)} rows (several series in
                        one file), so the analyses below use the average per timestamp.
                      </p>
                    )}
                    {fileId && (
                      <TimeSeriesAnalysis fileId={fileId} datetimeColumns={detectedCols} targets={data.targets ?? []} />
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
