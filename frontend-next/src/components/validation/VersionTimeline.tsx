"use client";

import { useWorkspaceStore } from "@/lib/store/workspace";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Clock } from "lucide-react";

interface VersionEntry {
  version: number;
  algorithm: string;
  numRows: number;
  timestamp: string;
  fileId: string;
}

export function VersionTimeline() {
  const generatedFileId = useWorkspaceStore((s) => s.generatedFileId);
  const selectedAlgorithm = useWorkspaceStore((s) => s.selectedAlgorithm);
  const generatorConfig = useWorkspaceStore((s) => s.generatorConfig);

  // Build a single-entry history from current state
  const history: VersionEntry[] = generatedFileId
    ? [
        {
          version: 1,
          algorithm: selectedAlgorithm ?? "Unknown",
          numRows: generatorConfig?.num_rows ?? 0,
          timestamp: new Date().toLocaleTimeString(),
          fileId: generatedFileId,
        },
      ]
    : [];

  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Clock className="h-4 w-4" />
          Generation History
        </CardTitle>
      </CardHeader>
      <CardContent>
        {history.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No generations yet. Use the Refinement panel to generate data.
          </p>
        ) : (
          <div className="space-y-2">
            {history.map((entry) => (
              <div
                key={entry.version}
                className="flex items-center justify-between rounded border px-3 py-2 text-sm"
              >
                <div className="flex items-center gap-3">
                  <span className="text-xs text-muted-foreground font-mono">
                    v{entry.version}
                  </span>
                  <div>
                    <span className="font-medium">{entry.algorithm}</span>
                    <span className="mx-1 text-muted-foreground">·</span>
                    <span className="text-muted-foreground">
                      {entry.numRows.toLocaleString()} rows
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">{entry.timestamp}</span>
                  <Badge variant="secondary" className="text-xs">
                    latest
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
