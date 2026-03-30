"use client";

import { useState } from "react";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
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

function VizFrame({ fileId, tab }: { fileId: string; tab: VizTab }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["viz", fileId, tab],
    queryFn: () =>
      apiClient
        .get<string>(`/visualize`, {
          params: { file_id: fileId, tab },
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
      className="w-full h-96 border rounded-md"
      title={`${tab} visualization`}
    />
  );
}

export function VizDashboard() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const [activeTab, setActiveTab] = useState<VizTab>("distribution");

  if (!fileId) return null;

  return (
    <Tabs
      value={activeTab}
      onValueChange={(v) => setActiveTab(v as VizTab)}
    >
      <TabsList className="flex-wrap h-auto gap-1">
        {VIZ_TABS.map((t) => (
          <TabsTrigger key={t.key} value={t.key} className="text-xs">
            {t.label}
          </TabsTrigger>
        ))}
      </TabsList>

      {/* P3.2: aria-live so screen readers announce tab content changes */}
      {VIZ_TABS.map((t) => (
        <TabsContent
          key={t.key}
          value={t.key}
          className="mt-3"
          aria-live="polite"
          aria-atomic="true"
        >
          <VizFrame fileId={fileId} tab={t.key} />
        </TabsContent>
      ))}
    </Tabs>
  );
}
