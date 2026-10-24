"use client";

import { useState } from "react";
import { useDataset } from "@/hooks/api/useDataset";
import { useCooksDistance } from "@/hooks/api/useAnalysis";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import type { AppError, CooksDistanceResponse } from "@/lib/api/types";

/** Influential points for a regression of one numeric column on others */
export function CooksDistanceSection() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const { data: dataset } = useDataset(fileId);
  const numeric = (dataset?.columns ?? []).filter((c) => c.type === "numeric").map((c) => c.name);
  const [target, setTarget] = useState<string | null>(null);
  const [features, setFeatures] = useState<string[] | null>(null);
  const cooks = useCooksDistance();
  const [result, setResult] = useState<CooksDistanceResponse["stats"][number] | null>(null);

  if (numeric.length < 2) return null;
  const tgt = target ?? numeric[numeric.length - 1];
  const feats = (features ?? numeric.filter((c) => c !== tgt)).filter((c) => c !== tgt);

  function run() {
    if (!fileId || feats.length === 0) return;
    cooks.mutate(
      { fileId, target: tgt, features: feats },
      {
        onSuccess: (r) => setResult(r ?? null),
        onError: (e) => toast.error("Cook's distance failed", { description: (e as Partial<AppError>).message }),
      }
    );
  }

  return (
    <div className="space-y-3 border-t pt-4">
      <div>
        <p className="text-sm font-medium">Influential points (Cook&apos;s distance)</p>
        <p className="text-xs text-muted-foreground">
          Rows that pull a linear regression of the target on the other columns the most.
        </p>
      </div>
      <div className="flex flex-wrap items-end gap-4">
        <div className="space-y-1">
          <Label htmlFor="cooks-target">Target</Label>
          <Select value={tgt} onValueChange={(v) => { setTarget(v as string); setResult(null); }}>
            <SelectTrigger id="cooks-target" className="w-72 max-w-full">
              <SelectValue className="truncate" />
            </SelectTrigger>
            <SelectContent>
              {numeric.map((c) => (
                <SelectItem key={c} value={c}>
                  {c}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <Label>Features</Label>
          <div className="flex flex-wrap gap-x-3 gap-y-1">
            {numeric
              .filter((c) => c !== tgt)
              .map((c) => (
                <div key={c} className="flex items-center gap-1.5">
                  <Checkbox
                    id={`cooks-${c}`}
                    checked={feats.includes(c)}
                    onCheckedChange={() =>
                      setFeatures(feats.includes(c) ? feats.filter((f) => f !== c) : [...feats, c])
                    }
                  />
                  <Label htmlFor={`cooks-${c}`} className="text-xs font-mono cursor-pointer">
                    {c}
                  </Label>
                </div>
              ))}
          </div>
        </div>
        <Button size="sm" variant="outline" onClick={run} disabled={cooks.isPending || feats.length === 0}>
          {cooks.isPending && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1" />}
          Find influential points
        </Button>
      </div>
      {result && (
        <p className="text-sm">
          <span className="font-medium">{result.outlier_count}</span> influential rows (
          {result.outlier_percent.toFixed(1)}%) above the threshold {result.threshold.toFixed(4)}; largest Cook&apos;s D ={" "}
          {result.max_cooks_d.toFixed(2)}.
        </p>
      )}
    </div>
  );
}
