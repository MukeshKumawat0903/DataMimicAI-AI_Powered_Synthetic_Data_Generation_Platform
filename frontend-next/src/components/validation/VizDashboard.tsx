"use client";

import { useState } from "react";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { useDataset } from "@/hooks/api/useDataset";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { apiClient } from "@/lib/api/client";
import { useQuery } from "@tanstack/react-query";

const VIZ_TABS = [
  { key: "distribution", label: "Distribution" },
  { key: "pairplot", label: "Pair Plot" },
  { key: "real_vs_synth", label: "Real vs Synth" },
  { key: "drift", label: "Drift" },
  { key: "correlation", label: "Correlation" },
] as const;

type VizTab = (typeof VIZ_TABS)[number]["key"];
type VizParams = Record<string, string | boolean | undefined>;

function VizFrame({ fileId, tab, params }: { fileId: string; tab: VizTab; params: VizParams }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["viz", fileId, tab, params],
    queryFn: () =>
      apiClient
        .get<string>(`/visualize`, {
          params: { file_id: fileId, tab, ...params },
          responseType: "text",
        })
        .then((r) => r.data),
    staleTime: 5 * 60 * 1000,
  });

  if (isLoading) return <Skeleton className="h-96 w-full" />;
  if (error)
    return (
      <p className="text-sm text-destructive p-4">
        Visualization failed — ensure synthetic data was generated successfully.
      </p>
    );

  return (
    <iframe
      srcDoc={data ?? ""}
      sandbox="allow-scripts allow-same-origin"
      className="w-full h-[28rem] border rounded-md"
      title={`${tab} visualization`}
    />
  );
}

function ColumnSelect({ id, label, value, options, onChange }: {
  id: string; label: string; value: string; options: string[]; onChange: (v: string) => void;
}) {
  return (
    <div className="space-y-1">
      <Label htmlFor={id}>{label}</Label>
      <Select value={value} onValueChange={(v) => onChange(v as string)}>
        <SelectTrigger id={id} className="w-52">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((o) => (
            <SelectItem key={o} value={o}>
              {o}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

function ColumnChecks({ idPrefix, options, picked, onChange }: {
  idPrefix: string; options: string[]; picked: string[]; onChange: (cols: string[]) => void;
}) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1">
      {options.map((c) => (
        <div key={c} className="flex items-center gap-1.5">
          <Checkbox
            id={`${idPrefix}-${c}`}
            checked={picked.includes(c)}
            onCheckedChange={() => onChange(picked.includes(c) ? picked.filter((p) => p !== c) : [...picked, c])}
          />
          <Label htmlFor={`${idPrefix}-${c}`} className="text-xs font-mono cursor-pointer">
            {c}
          </Label>
        </div>
      ))}
    </div>
  );
}

export function VizDashboard() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const { data: dataset } = useDataset(fileId);
  const [activeTab, setActiveTab] = useState<VizTab>("distribution");

  // Numeric columns first: most of these charts compare numeric distributions
  const cols = dataset?.columns ?? [];
  const numeric = cols.filter((c) => c.type === "numeric").map((c) => c.name);
  const ordered = [...numeric, ...cols.filter((c) => c.type !== "numeric").map((c) => c.name)];

  const [column, setColumn] = useState<string | null>(null);
  const [plotType, setPlotType] = useState("Histogram");
  const [overlay, setOverlay] = useState(true);
  const [pairX, setPairX] = useState<string | null>(null);
  const [pairY, setPairY] = useState<string | null>(null);
  const [compareCols, setCompareCols] = useState<string[] | null>(null);
  const [driftCols, setDriftCols] = useState<string[] | null>(null);
  const [showType, setShowType] = useState("Compare Both");

  if (!fileId) return null;
  if (!dataset) return <Skeleton className="h-96 w-full" />;

  const col = column ?? ordered[0];
  const px = pairX ?? ordered[0];
  const py = pairY ?? ordered[1] ?? ordered[0];
  const cmp = compareCols ?? numeric.slice(0, 2);
  const drift = driftCols ?? numeric.slice(0, 2);

  const params: Record<VizTab, VizParams> = {
    distribution: { column: col, plot_type: plotType, overlay },
    pairplot: { columns: `${px},${py}`, overlay },
    real_vs_synth: { columns: cmp.join(",") || undefined },
    drift: { columns: drift.join(",") || undefined },
    correlation: { show_type: showType },
  };

  return (
    <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as VizTab)}>
      <TabsList className="flex-wrap h-auto gap-1">
        {VIZ_TABS.map((t) => (
          <TabsTrigger key={t.key} value={t.key} className="text-xs">
            {t.label}
          </TabsTrigger>
        ))}
      </TabsList>

      {VIZ_TABS.map((t) => (
        <TabsContent key={t.key} value={t.key} className="mt-3 space-y-3" aria-live="polite" aria-atomic="true">
          <div className="flex flex-wrap items-end gap-4">
            {t.key === "distribution" && (
              <>
                <ColumnSelect id="viz-col" label="Column" value={col} options={ordered} onChange={setColumn} />
                <ColumnSelect id="viz-plot" label="Plot type" value={plotType} options={["Histogram", "KDE", "Boxplot"]} onChange={setPlotType} />
              </>
            )}
            {t.key === "pairplot" && (
              <>
                <ColumnSelect id="viz-x" label="X column" value={px} options={ordered} onChange={setPairX} />
                <ColumnSelect id="viz-y" label="Y column" value={py} options={ordered} onChange={setPairY} />
              </>
            )}
            {(t.key === "distribution" || t.key === "pairplot") && (
              <div className="flex items-center gap-1.5 pb-2">
                <Checkbox id={`viz-overlay-${t.key}`} checked={overlay} onCheckedChange={() => setOverlay((o) => !o)} />
                <Label htmlFor={`viz-overlay-${t.key}`} className="text-sm cursor-pointer">Overlay real and synthetic</Label>
              </div>
            )}
            {t.key === "real_vs_synth" && <ColumnChecks idPrefix="viz-cmp" options={ordered} picked={cmp} onChange={setCompareCols} />}
            {t.key === "drift" && <ColumnChecks idPrefix="viz-drift" options={numeric} picked={drift} onChange={setDriftCols} />}
            {t.key === "correlation" && (
              <ColumnSelect id="viz-show" label="Show" value={showType} options={["Compare Both", "Real Data", "Synthetic Data"]} onChange={setShowType} />
            )}
          </div>
          <VizFrame fileId={fileId} tab={t.key} params={params[t.key]} />
        </TabsContent>
      ))}
    </Tabs>
  );
}
