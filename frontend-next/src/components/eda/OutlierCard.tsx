"use client";

import { useState } from "react";
import { useOutliers, useRemediateOutliers } from "@/hooks/api/useOutliers";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ChevronDown, ChevronRight } from "lucide-react";
import { toast } from "sonner";
import type { OutlierResult } from "@/lib/api/types";

const DETECTION_METHODS = ["IQR", "ZScore", "IsolationForest", "LOF"] as const;
type DetectionMethod = (typeof DETECTION_METHODS)[number];
type RemediationAction = "remove" | "cap" | "transform" | "impute";

export function OutlierCard() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const [open, setOpen] = useState(false);
  const [triggered, setTriggered] = useState(false);
  const [methods, setMethods] = useState<DetectionMethod[]>(["IQR"]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [remediations, setRemediations] = useState<
    Record<string, RemediationAction>
  >({});

  const { data, isLoading, error, refetch } = useOutliers(
    triggered ? fileId : null,
    methods
  );
  const remediate = useRemediateOutliers();

  function handleRun() {
    setTriggered(true);
    setOpen(true);
    if (triggered) refetch();
  }

  function toggleMethod(m: DetectionMethod) {
    setMethods((prev) =>
      prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]
    );
  }

  function handleRemediate() {
    if (!fileId) return;
    remediate.mutate(
      {
        file_id: fileId,
        remediations: Object.entries(remediations).map(([column, action]) => ({
          column,
          action,
        })),
      },
      {
        onSuccess: () => {
          setDialogOpen(false);
          toast.success("Outliers remediated", { description: "Dataset has been updated." });
        },
        onError: () => toast.error("Remediation failed", { description: "Please try again." }),
      }
    );
  }

  const affectedColumns = data
    ? Array.from(new Set(data.results.map((r: OutlierResult) => r.column)))
    : [];

  return (
    <>
      <Collapsible open={open} onOpenChange={setOpen}>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between py-3">
            <CollapsibleTrigger
              className="flex items-center gap-2 text-sm font-medium"
            >
              {open ? (
                  <ChevronDown className="h-4 w-4" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
              <CardTitle className="text-base">Outlier Detection</CardTitle>
            </CollapsibleTrigger>
            <div className="flex items-center gap-2">
              {data && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setDialogOpen(true)}
                >
                  Remediate
                </Button>
              )}
              <Button size="sm" variant="outline" onClick={handleRun}>
                {triggered ? "Re-run" : "Run Detection"}
              </Button>
            </div>
          </CardHeader>

          <CollapsibleContent>
            <CardContent className="pt-0 space-y-3">
              <div className="flex flex-wrap gap-3">
                {DETECTION_METHODS.map((m) => (
                  <div key={m} className="flex items-center gap-1.5">
                    <Checkbox
                      id={`method-${m}`}
                      checked={methods.includes(m)}
                      onCheckedChange={() => toggleMethod(m)}
                    />
                    <Label htmlFor={`method-${m}`} className="text-sm cursor-pointer">
                      {m}
                    </Label>
                  </div>
                ))}
              </div>

              {isLoading && (
                <div className="space-y-2">
                  {[...Array(4)].map((_, i) => (
                    <Skeleton key={i} className="h-8 w-full" />
                  ))}
                </div>
              )}
              {error && (
                <p className="text-sm text-destructive">
                  Detection failed. Please try again.
                </p>
              )}
              {data && (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Column</TableHead>
                      <TableHead>Method</TableHead>
                      <TableHead>Outlier Count</TableHead>
                      <TableHead>% Affected</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.results.map((r: OutlierResult, i: number) => (
                      <TableRow key={i}>
                        <TableCell className="font-mono text-xs">{r.column}</TableCell>
                        <TableCell>
                          <Badge variant="outline" className="text-xs">
                            {r.method}
                          </Badge>
                        </TableCell>
                        <TableCell>{r.outlier_count}</TableCell>
                        <TableCell>
                          <span
                            className={
                              r.pct_affected > 10
                                ? "text-destructive font-medium"
                                : r.pct_affected > 5
                                ? "text-amber-600"
                                : ""
                            }
                          >
                            {r.pct_affected.toFixed(1)}%
                          </span>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Remediate Outliers</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            {affectedColumns.map((col) => (
              <div key={col} className="flex items-center justify-between gap-2">
                <span className="font-mono text-sm">{col}</span>
                <select
                  className="border rounded px-2 py-1 text-sm bg-background"
                  value={remediations[col] ?? "cap"}
                  onChange={(e) =>
                    setRemediations((prev) => ({
                      ...prev,
                      [col]: e.target.value as RemediationAction,
                    }))
                  }
                  aria-label={`Remediation action for ${col}`}
                >
                  <option value="cap">Cap (Winsorize)</option>
                  <option value="remove">Remove rows</option>
                  <option value="transform">Log transform</option>
                  <option value="impute">Impute median</option>
                </select>
              </div>
            ))}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleRemediate}
              disabled={remediate.isPending}
            >
              {remediate.isPending ? "Applying…" : "Apply"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
