"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ArrowRight } from "lucide-react";
import Link from "next/link";

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
const LLMExplanationPanel = dynamic(
  () =>
    import("@/components/eda/LLMExplanationPanel").then(
      (m) => m.LLMExplanationPanel
    ),
  { loading: cardSkeleton }
);

export default function ExplorePage() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const router = useRouter();

  // Guard: redirect to upload if no file loaded
  useEffect(() => {
    if (!fileId) {
      router.replace("/upload");
    }
  }, [fileId, router]);

  if (!fileId) return null;

  return (
    <div className="max-w-4xl mx-auto space-y-4 p-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Explore Data</h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            Run each analysis independently — results are cached for 5 minutes.
          </p>
        </div>
        <Link href="/upload">
          <Button variant="ghost" size="sm" className="gap-1">
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
        </Link>
      </div>

      <div className="space-y-3">
        <ProfilingCard />
        <CorrelationCard />
        <OutlierCard />
        <PrivacyAuditCard />
        <FeatureSuggestionsCard />
        {/* P0.4: Time-series analysis card */}
        <TimeSeriesCard />
      </div>

      <div className="border rounded-lg p-4 bg-muted/30">
        <LLMExplanationPanel />
      </div>

      <div className="flex justify-end">
        <Link href="/generate">
          <Button className="gap-1">
            Continue to Generate
            <ArrowRight className="h-4 w-4" />
          </Button>
        </Link>
      </div>
    </div>
  );
}
