"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { GenerationPanel } from "@/components/generation/GenerationPanel";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ArrowRight } from "lucide-react";
import Link from "next/link";

export default function GeneratePage() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const generatedFileId = useWorkspaceStore((s) => s.generatedFileId);
  const router = useRouter();

  useEffect(() => {
    if (!fileId) {
      router.replace("/upload");
    }
  }, [fileId, router]);

  if (!fileId) return null;

  return (
    <div className="max-w-2xl mx-auto space-y-4 p-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Generate Synthetic Data</h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            Choose an algorithm, configure parameters, and generate.
          </p>
        </div>
        <Link href="/explore">
          <Button variant="ghost" size="sm" className="gap-1">
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
        </Link>
      </div>

      <GenerationPanel />

      {generatedFileId && (
        <div className="flex justify-end">
          <Link href="/validate">
            <Button className="gap-1">
              Continue to Validate
              <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
        </div>
      )}
    </div>
  );
}
