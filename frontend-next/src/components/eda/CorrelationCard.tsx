"use client";

import { useState } from "react";
import { useCorrelation } from "@/hooks/api/useCorrelation";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ChevronDown, ChevronRight, AlertTriangle } from "lucide-react";

function correlationColor(v: number): string {
  const abs = Math.abs(v);
  if (abs > 0.99) return "text-destructive font-bold";
  if (abs > 0.8) return "text-red-600 font-medium";
  if (abs > 0.5) return "text-amber-600";
  return "";
}

export function CorrelationCard() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const [open, setOpen] = useState(false);
  const [triggered, setTriggered] = useState(false);

  const { data, isLoading, error, refetch } = useCorrelation(
    triggered ? fileId : null
  );

  function handleRun() {
    setTriggered(true);
    setOpen(true);
    if (triggered) refetch();
  }

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between py-3">
          <CollapsibleTrigger
              className="flex items-center gap-2 text-sm font-medium"
            >
              {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
              <CardTitle className="text-base">Correlation Analysis</CardTitle>
            </CollapsibleTrigger>
          <Button size="sm" variant="outline" onClick={handleRun}>
            {triggered ? "Re-run" : "Run Correlation"}
          </Button>
        </CardHeader>

        <CollapsibleContent>
          <CardContent className="pt-0 space-y-4">
            {/* P1.2: Hint text before the user triggers analysis */}
            {!triggered && !isLoading && !data && !error && (
              <p className="text-sm text-muted-foreground">
                Click &quot;Run Correlation&quot; to detect high-correlation column pairs that may affect synthesis quality.
              </p>
            )}
            {isLoading && (
              <div className="space-y-2">
                <Skeleton className="h-64 w-full" />
                <Skeleton className="h-8 w-full" />
                <Skeleton className="h-8 w-full" />
              </div>
            )}
            {error && (
              <p className="text-sm text-destructive">
                Failed to compute correlation. Please try again.
              </p>
            )}
            {data && (
              <>
                {data.heatmap_base64 && (
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Pearson Correlation Heatmap</p>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`data:image/png;base64,${data.heatmap_base64}`}
                      alt="Correlation heatmap"
                      className="w-full rounded border"
                    />
                  </div>
                )}
                <div>
                  <p className="text-sm font-medium mb-2">Top Correlated Pairs</p>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Column A</TableHead>
                        <TableHead>Column B</TableHead>
                        <TableHead>Correlation</TableHead>
                        <TableHead>Flag</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {data.top_pairs.map((pair, i) => (
                        <TableRow key={i}>
                          <TableCell className="font-mono text-xs">{pair.col_a}</TableCell>
                          <TableCell className="font-mono text-xs">{pair.col_b}</TableCell>
                          <TableCell className={`text-sm ${correlationColor(pair.correlation)}`}>
                            {pair.correlation.toFixed(3)}
                          </TableCell>
                          <TableCell>
                            {pair.is_leakage && (
                              <Badge variant="destructive" className="text-xs flex items-center gap-1 w-fit">
                                <AlertTriangle className="h-3 w-3" />
                                Data Leakage
                              </Badge>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </>
            )}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
