"use client";

import { useState } from "react";
import { useProfile } from "@/hooks/api/useProfile";
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
import { ChevronDown, ChevronRight, Lock, AlertTriangle } from "lucide-react";

export function ProfilingCard() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const [open, setOpen] = useState(false);
  const [triggered, setTriggered] = useState(false);

  const { data, isLoading, error, refetch } = useProfile(
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
              {open ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
              <CardTitle className="text-base">Data Profiling</CardTitle>
            </CollapsibleTrigger>
          <Button size="sm" variant="outline" onClick={handleRun}>
            {triggered ? "Re-run" : "Run Profile"}
          </Button>
        </CardHeader>

        <CollapsibleContent>
          <CardContent className="pt-0">
            {isLoading && (
              <div className="space-y-2">
                {[...Array(5)].map((_, i) => (
                  <Skeleton key={i} className="h-8 w-full" />
                ))}
              </div>
            )}
            {error && (
              <p className="text-sm text-destructive">
                Failed to load profile. Please try again.
              </p>
            )}
            {data && (
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Column</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Missing %</TableHead>
                      <TableHead>Unique</TableHead>
                      <TableHead>Suggestions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.columns.map((col) => (
                      <TableRow key={col.column}>
                        <TableCell className="font-mono text-xs">
                          <span className="flex items-center gap-1">
                            {col.is_pii && (
                              <Lock className="h-3 w-3 text-amber-500" aria-label="PII column" />
                            )}
                            {col.column}
                          </span>
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline" className="text-xs">
                            {col.dtype}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <span
                            className={
                              col.missing_pct > 20
                                ? "text-destructive font-medium"
                                : col.missing_pct > 5
                                ? "text-amber-600 font-medium"
                                : ""
                            }
                          >
                            {col.missing_pct.toFixed(1)}%
                          </span>
                        </TableCell>
                        <TableCell>{col.unique_count}</TableCell>
                        <TableCell>
                          <div className="flex flex-wrap gap-1">
                            {col.suggestions.map((s) => (
                              <Badge key={s} variant="secondary" className="text-xs">
                                {s}
                              </Badge>
                            ))}
                            {col.missing_pct > 20 && (
                              <AlertTriangle className="h-3.5 w-3.5 text-amber-500 inline" />
                            )}
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                <p className="mt-2 text-xs text-muted-foreground">
                  {data.row_count.toLocaleString()} rows analysed
                </p>
              </div>
            )}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
