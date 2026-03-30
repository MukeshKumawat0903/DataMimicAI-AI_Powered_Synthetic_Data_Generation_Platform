"use client";

import { useState } from "react";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { useGenerate } from "@/hooks/api/useGenerate";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { RefreshCw } from "lucide-react";

export function RefinementPanel() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const selectedAlgorithm = useWorkspaceStore((s) => s.selectedAlgorithm);
  const [numRows, setNumRows] = useState(1000);
  const generate = useGenerate();

  function handleRegenerate() {
    if (!fileId) return;
    generate.mutate({
      file_id: fileId,
      feedback: [],
      generator_config: {
        algorithm: selectedAlgorithm ?? "GaussianCopula",
        num_rows: numRows,
      },
    });
  }

  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="text-base">Iterative Refinement</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Adjust parameters and regenerate synthetic data to improve quality scores.
        </p>

        <div className="space-y-2">
          <Label className="text-sm">
            Rows to generate: <span className="font-mono font-medium">{numRows.toLocaleString()}</span>
          </Label>
          <input
            type="range"
            min={100}
            max={10000}
            step={100}
            value={numRows}
            onChange={(e) => setNumRows(Number(e.target.value))}
            className="w-full accent-primary"
            aria-label="Number of rows to generate"
          />
        </div>

        <Button
          onClick={handleRegenerate}
          disabled={!fileId || generate.isPending}
          className="gap-2"
        >
          <RefreshCw className={`h-4 w-4 ${generate.isPending ? "animate-spin" : ""}`} />
          {generate.isPending ? "Regenerating…" : "Regenerate"}
        </Button>

        {generate.isError && (
          <p className="text-sm text-destructive">
            Generation failed. Please try again.
          </p>
        )}
        {generate.isSuccess && (
          <p className="text-sm text-green-600 dark:text-green-400">
            New synthetic data generated. Quality scores updated above.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
