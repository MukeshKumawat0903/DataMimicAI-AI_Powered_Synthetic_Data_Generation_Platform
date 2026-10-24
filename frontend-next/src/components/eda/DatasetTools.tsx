"use client";

import { useState } from "react";
import { useDataset } from "@/hooks/api/useDataset";
import { useResetDataset } from "@/hooks/api/useAnalysis";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { apiClient } from "@/lib/api/client";
import { downloadBlob } from "@/lib/download";
import { DataTable } from "@/components/data/DataTable";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDown, ChevronRight, Download, RotateCcw, Table2 } from "lucide-react";
import { IconTile } from "@/components/ui/icon-tile";
import { toast } from "sonner";
import type { AppError } from "@/lib/api/types";

/** Download the working dataset, or reset it to the original upload */
export function DatasetActions() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const reset = useResetDataset();
  const [confirm, setConfirm] = useState(false);
  if (!fileId) return null;

  async function download() {
    try {
      const res = await apiClient.get(`/eda/download`, { params: { file_id: fileId }, responseType: "blob" });
      downloadBlob(res.data as Blob, "dataset.csv");
    } catch (e) {
      toast.error("Download failed", { description: (e as Partial<AppError>).message });
    }
  }

  return (
    <div className="flex items-center gap-1">
      <Button variant="ghost" size="sm" className="gap-1" onClick={download}>
        <Download className="h-4 w-4" /> Download dataset
      </Button>
      {confirm ? (
        <Button
          variant="destructive"
          size="sm"
          className="gap-1"
          disabled={reset.isPending}
          onClick={() =>
            reset.mutate(fileId, {
              onSuccess: (r) => {
                setConfirm(false);
                toast.success("Dataset reset", { description: r.message });
              },
              onError: (e) => toast.error("Reset failed", { description: (e as Partial<AppError>).message }),
            })
          }
        >
          <RotateCcw className="h-4 w-4" /> Confirm reset
        </Button>
      ) : (
        <Button variant="ghost" size="sm" className="gap-1" onClick={() => setConfirm(true)}>
          <RotateCcw className="h-4 w-4" /> Reset to original
        </Button>
      )}
    </div>
  );
}

/** The working dataset, with columns changed by the last cleaning action highlighted */
export function CurrentDatasetPanel() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const changed = useWorkspaceStore((s) => s.lastChangedColumns);
  const [open, setOpen] = useState(false);
  const { data } = useDataset(open ? fileId : null);

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <Card>
        <CardHeader className="py-3">
          <CollapsibleTrigger className="flex items-center gap-3 text-sm font-medium">
            {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            <IconTile icon={Table2} tone="slate" />
            <span className="text-left">
              <CardTitle className="text-base">Current dataset</CardTitle>
              <span className="block text-xs font-normal text-muted-foreground">The working data, with recently changed columns highlighted</span>
            </span>
          </CollapsibleTrigger>
        </CardHeader>
        <CollapsibleContent>
          <CardContent className="pt-0 space-y-2">
            {changed.length > 0 && (
              <p className="text-xs text-amber-700">
                Highlighted: changed by the last action (<span className="font-mono">{changed.join(", ")}</span>)
              </p>
            )}
            {data && (
              <>
                <p className="text-xs text-muted-foreground">
                  {data.num_rows.toLocaleString()} rows × {data.num_columns} columns
                </p>
                <DataTable data={data.sample_data} columnInfo={data.columns} pageSize={10} highlightColumns={changed} />
              </>
            )}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
