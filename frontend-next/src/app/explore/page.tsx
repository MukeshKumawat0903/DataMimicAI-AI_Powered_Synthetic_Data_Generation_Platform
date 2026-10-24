"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import { useStoreHydrated } from "@/lib/store/useStoreHydrated";
import { useWorkspaceStore, type ExploreMode } from "@/lib/store/workspace";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ArrowLeft, ArrowRight, BarChart3, Bot, Network, Radar, ScanSearch, ShieldCheck, Sparkles, Table2, TrendingUp } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { CurrentDatasetPanel, DatasetActions } from "@/components/eda/DatasetTools";
import Link from "next/link";

// Icon colours match the tile in each card's header
const ANALYSIS_TABS = [
  { value: "profiling", label: "Profiling", icon: ScanSearch, color: "text-sky-500" },
  { value: "correlation", label: "Correlation", icon: Network, color: "text-violet-500" },
  { value: "outliers", label: "Outliers", icon: Radar, color: "text-orange-500" },
  { value: "privacy", label: "Privacy", icon: ShieldCheck, color: "text-emerald-500" },
  { value: "features", label: "Feature Suggestions", icon: Sparkles, color: "text-fuchsia-500" },
  { value: "timeseries", label: "Time Series", icon: TrendingUp, color: "text-cyan-500" },
  { value: "dataset", label: "Dataset", icon: Table2, color: "text-slate-500" },
] as const;

// P2.3: Lazy-load all EDA cards to split the bundle
const cardSkeleton = () => <Skeleton className="h-12 w-full rounded-lg" />;

const ProfilingCard = dynamic(
  () => import("@/components/eda/ProfilingCard").then((m) => m.ProfilingCard),
  { loading: cardSkeleton }
);
const CorrelationCard = dynamic(
  () => import("@/components/eda/CorrelationCard").then((m) => m.CorrelationCard),
  { loading: cardSkeleton }
);
const OutlierCard = dynamic(
  () => import("@/components/eda/OutlierCard").then((m) => m.OutlierCard),
  { loading: cardSkeleton }
);
const PrivacyAuditCard = dynamic(
  () => import("@/components/eda/PrivacyAuditCard").then((m) => m.PrivacyAuditCard),
  { loading: cardSkeleton }
);
const FeatureSuggestionsCard = dynamic(
  () =>
    import("@/components/eda/FeatureSuggestionsCard").then(
      (m) => m.FeatureSuggestionsCard
    ),
  { loading: cardSkeleton }
);
const TimeSeriesCard = dynamic(
  () =>
    import("@/components/eda/TimeSeriesCard").then((m) => m.TimeSeriesCard),
  { loading: cardSkeleton }
);
const AIAssistance = dynamic(
  () => import("@/components/ai/AIAssistance").then((m) => m.AIAssistance),
  { loading: cardSkeleton }
);

export default function ExplorePage() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const exploreMode = useWorkspaceStore((s) => s.exploreMode);
  const setExploreMode = useWorkspaceStore((s) => s.setExploreMode);
  const router = useRouter();

  // Guard: redirect to upload if no file loaded
  // Wait for the saved session to load before deciding there's no dataset
  const hydrated = useStoreHydrated();
  useEffect(() => {
    if (hydrated && !fileId) {
      router.replace("/upload");
    }
  }, [hydrated, fileId, router]);

  if (!fileId) return null;

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-6 py-8">
      <PageHeader
        icon={BarChart3}
        step="Step 2 of 4"
        title="Explore Data"
        description="Run each analysis independently — results are cached for 5 minutes."
        actions={
          <>
            <DatasetActions />
            <Link href="/upload">
              <Button variant="ghost" size="sm" className="gap-1">
                <ArrowLeft className="h-4 w-4" />
                Back
              </Button>
            </Link>
          </>
        }
      />

      <Tabs value={exploreMode} onValueChange={(v) => setExploreMode(v as ExploreMode)}>
        <TabsList className="h-11 p-1">
          <TabsTrigger value="analysis" className="gap-2 px-4">
            <BarChart3 className="h-4 w-4" />
            Data Analysis
          </TabsTrigger>
          <TabsTrigger value="ai" className="gap-2 px-4">
            <Bot className="h-4 w-4" />
            AI Assistance
          </TabsTrigger>
        </TabsList>

        {/* Both modes stay mounted so switching between them keeps every result */}
        <TabsContent value="analysis" keepMounted className="mt-3">
          {/* One analysis at a time, like the Streamlit app; panels stay mounted so results survive tab switches */}
          <Tabs defaultValue="profiling">
            <TabsList className="h-auto flex-wrap gap-1 bg-card p-1.5 shadow-sm">
              {ANALYSIS_TABS.map(({ value, label, icon: Icon, color }) => (
                <TabsTrigger key={value} value={value} className="h-8 gap-1.5 text-xs data-active:shadow-md">
                  <Icon className={`h-3.5 w-3.5 ${color}`} />
                  {label}
                </TabsTrigger>
              ))}
            </TabsList>
            <TabsContent value="profiling" keepMounted className="mt-3"><ProfilingCard /></TabsContent>
            <TabsContent value="correlation" keepMounted className="mt-3"><CorrelationCard /></TabsContent>
            <TabsContent value="outliers" keepMounted className="mt-3"><OutlierCard /></TabsContent>
            <TabsContent value="privacy" keepMounted className="mt-3"><PrivacyAuditCard /></TabsContent>
            <TabsContent value="features" keepMounted className="mt-3"><FeatureSuggestionsCard /></TabsContent>
            <TabsContent value="timeseries" keepMounted className="mt-3"><TimeSeriesCard /></TabsContent>
            <TabsContent value="dataset" keepMounted className="mt-3"><CurrentDatasetPanel /></TabsContent>
          </Tabs>
        </TabsContent>

        <TabsContent value="ai" keepMounted className="mt-3">
          <AIAssistance />
        </TabsContent>
      </Tabs>

      <div className="flex justify-end">
        <Link href="/generate">
          <Button size="lg" className="gap-2">
            Continue to Generate
            <ArrowRight className="h-4 w-4" />
          </Button>
        </Link>
      </div>
    </div>
  );
}
