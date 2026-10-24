"use client";

import { useState } from "react";
import { useEvaluateModels, type ModelEvaluation } from "@/hooks/api/useGenerate";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { SYNTHCITY_MODEL_CODES } from "@/lib/constants";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Loader2, Trophy } from "lucide-react";
import { IconTile } from "@/components/ui/icon-tile";
import { toast } from "sonner";
import type { Algorithm, AppError } from "@/lib/api/types";

const fmt = (v: number | null) => (v == null ? "—" : v.toFixed(3));

/** Train every SynthCity model briefly and rank them by distribution similarity */
export function AutoMLPanel() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const setSelectedAlgorithm = useWorkspaceStore((s) => s.setSelectedAlgorithm);
  const evaluate = useEvaluateModels();
  const [epochs, setEpochs] = useState(50);
  const [result, setResult] = useState<ModelEvaluation | null>(null);
  const bestCode = result ? SYNTHCITY_MODEL_CODES[result.best_model] : undefined;

  return (
    <div className="space-y-3 rounded-xl border bg-gradient-to-br from-amber-50/80 to-card p-4 dark:from-amber-500/10 dark:to-card">
      <div className="flex items-start gap-3">
        <IconTile icon={Trophy} tone="amber" />
        <div>
          <p className="text-sm font-semibold">AutoML: find the best model</p>
          <p className="text-xs text-muted-foreground">
            Trains all 7 SynthCity models briefly and compares their output with your data. It needs plenty of free
            memory and can take several minutes; models are skipped when memory is low.
          </p>
        </div>
      </div>
      <div className="flex items-end gap-2">
        <div className="space-y-1">
          <Label htmlFor="automl-epochs">Epochs per model</Label>
          <Input id="automl-epochs" type="number" min={5} className="h-8 w-24" value={epochs}
            onChange={(e) => setEpochs(Math.max(5, Number(e.target.value) || 5))} />
        </div>
        <Button size="sm" variant="outline" disabled={!fileId || evaluate.isPending}
          onClick={() =>
            fileId &&
            evaluate.mutate({ fileId, epochs }, {
              onSuccess: setResult,
              onError: (e) => toast.error("Model comparison failed", { description: (e as Partial<AppError>).message }),
            })
          }>
          {evaluate.isPending && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1" />}
          {evaluate.isPending ? "Comparing models…" : "Compare models"}
        </Button>
      </div>

      {result && (
        <>
          <p className="text-sm">
            {result.summary.successful_models} of {result.summary.total_models} models finished in{" "}
            {result.summary.total_evaluation_time}s.{" "}
            {bestCode ? (
              <span className="inline-flex items-center gap-1 font-medium">
                <Trophy className="h-3.5 w-3.5 text-amber-500" /> Best: {result.best_model}
              </span>
            ) : (
              <span className="text-amber-700">{result.best_model}</span>
            )}
          </p>
          {bestCode && (
            <Button size="sm" onClick={() => { setSelectedAlgorithm(bestCode as Algorithm); toast.success(`Selected ${bestCode}`); }}>
              Use this model
            </Button>
          )}
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">KS</TableHead>
                <TableHead className="text-right">JS</TableHead>
                <TableHead className="text-right">Wasserstein</TableHead>
                <TableHead className="text-right">MMD</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {result.metrics.map((m) => (
                <TableRow key={m.Model}>
                  <TableCell className="text-xs">{m.Model.replace("SynthCity - ", "")}</TableCell>
                  <TableCell className="text-xs">
                    {m.Status}
                    {m.Error && <span className="block text-muted-foreground">{m.Error}</span>}
                  </TableCell>
                  <TableCell className="text-right text-xs">{fmt(m.KolmogorovSmirnov)}</TableCell>
                  <TableCell className="text-right text-xs">{fmt(m.JensenShannon)}</TableCell>
                  <TableCell className="text-right text-xs">{fmt(m.Wasserstein)}</TableCell>
                  <TableCell className="text-right text-xs">{fmt(m.MaxMeanDiscrep)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </>
      )}
    </div>
  );
}
