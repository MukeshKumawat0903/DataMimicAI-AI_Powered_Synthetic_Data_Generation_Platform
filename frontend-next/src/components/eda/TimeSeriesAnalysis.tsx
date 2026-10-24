"use client";

import { useState } from "react";
import { useAcfPacf, useDecomposition, useTimeseriesOutliers, type TimeseriesOutliers } from "@/hooks/api/useAnalysis";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Bar, CartesianGrid, ComposedChart, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import type { AcfPacfResponse, AppError, DecompositionResponse } from "@/lib/api/types";

const MAX_POINTS = 400;

/** Every n-th point, so long series stay fast to draw */
function downsample(values: (number | null)[], index: string[]) {
  const step = Math.max(1, Math.ceil(values.length / MAX_POINTS));
  const out: { t: string; v: number | null }[] = [];
  for (let i = 0; i < values.length; i += step) out.push({ t: index[i]?.slice(0, 10) ?? String(i), v: values[i] });
  return out;
}

function Correlogram({ title, data }: { title: string; data: AcfPacfResponse["acf_pacf"]["acf"] }) {
  // Lag 0 is always 1; the band is the confidence interval half-width drawn around zero
  const rows = data.lags.slice(1).map((lag, i) => {
    const half = data.upper[i + 1] - data.values[i + 1];
    return { lag, value: data.values[i + 1], band: half, negBand: -half };
  });
  return (
    <div>
      <p className="text-xs text-muted-foreground mb-1">{title}</p>
      <div className="h-44">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="lag" fontSize={11} />
            <YAxis fontSize={11} domain={[-1, 1]} />
            <Tooltip formatter={(v) => Number(v).toFixed(3)} />
            <Bar dataKey="value" fill="#2563eb" name="Correlation" />
            <Line dataKey="band" stroke="#94a3b8" strokeDasharray="4 3" dot={false} name="95% band" />
            <Line dataKey="negBand" stroke="#94a3b8" strokeDasharray="4 3" dot={false} legendType="none" name="95% band" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Panel({ title, series }: { title: string; series: { values: (number | null)[]; index: string[] } }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground mb-1">{title}</p>
      <div className="h-28">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={downsample(series.values, series.index)}>
            <XAxis dataKey="t" hide />
            <YAxis fontSize={10} width={48} />
            <Tooltip />
            <Line dataKey="v" stroke="#2563eb" dot={false} strokeWidth={1.2} name={title} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function TimeSeriesAnalysis({
  fileId,
  datetimeColumns,
  targets,
}: {
  fileId: string;
  datetimeColumns: string[];
  targets: string[];
}) {
  const [dt, setDt] = useState(datetimeColumns[0]);
  const [target, setTarget] = useState(targets[0]);
  const acf = useAcfPacf();
  const decompose = useDecomposition();
  const [acfResult, setAcfResult] = useState<AcfPacfResponse["acf_pacf"] | null>(null);
  const [dec, setDec] = useState<DecompositionResponse["decomposition"] | null>(null);
  const tsOutliers = useTimeseriesOutliers();
  const [outl, setOutl] = useState<TimeseriesOutliers | null>(null);
  const failed = (what: string) => (e: unknown) => toast.error(`${what} failed`, { description: (e as Partial<AppError>).message });

  if (!targets.length) {
    return <p className="text-sm text-muted-foreground">No numeric column to analyse over time.</p>;
  }
  const input = { fileId, datetimeCol: dt, targetCol: target };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <div className="space-y-1">
          <Label htmlFor="ts-date">Date column</Label>
          <Select value={dt} onValueChange={(v) => setDt(v as string)}>
            <SelectTrigger id="ts-date" className="w-44">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {datetimeColumns.map((c) => (
                <SelectItem key={c} value={c}>
                  {c}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <Label htmlFor="ts-target">Value column</Label>
          <Select value={target} onValueChange={(v) => setTarget(v as string)}>
            <SelectTrigger id="ts-target" className="w-44">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {targets.map((c) => (
                <SelectItem key={c} value={c}>
                  {c}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <Button size="sm" variant="outline" disabled={acf.isPending}
          onClick={() => acf.mutate(input, { onSuccess: setAcfResult, onError: failed("ACF/PACF") })}>
          {acf.isPending && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1" />}
          ACF / PACF
        </Button>
        <Button size="sm" variant="outline" disabled={decompose.isPending}
          onClick={() => decompose.mutate(input, { onSuccess: setDec, onError: failed("Decomposition") })}>
          {decompose.isPending && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1" />}
          Decompose
        </Button>
        <Button size="sm" variant="outline" disabled={tsOutliers.isPending}
          onClick={() => tsOutliers.mutate(input, { onSuccess: setOutl, onError: failed("Time-series outlier detection") })}>
          {tsOutliers.isPending && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1" />}
          Find outliers
        </Button>
      </div>

      {outl?.stats[0] && (
        <p className="text-sm">
          <span className="font-medium">{outl.stats[0].outlier_count}</span> unusual points (
          {outl.stats[0].outlier_percent.toFixed(1)}%) stand out from the trend and seasonality
          {(outl.outlier_indices[target] ?? []).length > 0 && (
            <span className="text-muted-foreground">
              : {(outl.outlier_indices[target] ?? []).slice(0, 8).map((d) => d.slice(0, 10)).join(", ")}
              {(outl.outlier_indices[target] ?? []).length > 8 ? "…" : ""}
            </span>
          )}
          .
        </p>
      )}

      {acfResult && (
        <div className="space-y-2">
          <div className="grid gap-4 md:grid-cols-2">
            <Correlogram title="Autocorrelation (ACF)" data={acfResult.acf} />
            <Correlogram title="Partial autocorrelation (PACF)" data={acfResult.pacf} />
          </div>
          <p className="text-xs text-muted-foreground">
            Suggested model: <span className="font-medium">{acfResult.suggestions.model_type}</span> —{" "}
            {acfResult.suggestions.explanation} ({acfResult.n_obs.toLocaleString()} observations)
          </p>
        </div>
      )}

      {dec && (
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground">
            {dec.model} decomposition, period {dec.period} · trend strength {dec.strength_of_trend.toFixed(2)} · seasonality
            strength {dec.strength_of_seasonality.toFixed(2)}
          </p>
          {/* Block layout: an auto-sized grid column would shrink-wrap the responsive charts */}
          <div className="space-y-2">
            <Panel title="Observed" series={dec.observed} />
            <Panel title="Trend" series={dec.trend} />
            <Panel title="Seasonal" series={dec.seasonal} />
            <Panel title="Residual" series={dec.residual} />
          </div>
        </div>
      )}
    </div>
  );
}
