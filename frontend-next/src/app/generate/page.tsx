"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useStoreHydrated } from "@/lib/store/useStoreHydrated";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { GenerationPanel } from "@/components/generation/GenerationPanel";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ArrowRight, Wand2 } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import Link from "next/link";

export default function GeneratePage() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const generatedFileId = useWorkspaceStore((s) => s.generatedFileId);
  const router = useRouter();

  // Wait for the saved session to load before deciding there's no dataset
  const hydrated = useStoreHydrated();
  useEffect(() => {
    if (hydrated && !fileId) {
      router.replace("/upload");
    }
  }, [hydrated, fileId, router]);

  if (!fileId) return null;

  return (
    <div className="mx-auto max-w-4xl space-y-6 px-6 py-8">
      <PageHeader
        icon={Wand2}
        step="Step 3 of 4"
        title="Generate Synthetic Data"
        description="Choose an algorithm, configure parameters, and generate."
        actions={
          <Link href="/explore">
            <Button variant="ghost" size="sm" className="gap-1">
              <ArrowLeft className="h-4 w-4" />
              Back
            </Button>
          </Link>
        }
      />

      <GenerationPanel />

      {generatedFileId && (
        <div className="flex justify-end">
          <Link href="/validate">
            <Button size="lg" className="gap-2">
              Continue to Validate
              <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
        </div>
      )}
    </div>
  );
}
