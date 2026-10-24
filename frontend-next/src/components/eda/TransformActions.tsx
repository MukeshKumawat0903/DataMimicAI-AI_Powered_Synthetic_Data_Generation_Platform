"use client";

import { useState } from "react";
import {
  decodeTransformedData,
  useApplyTransforms,
  useLoadTransformConfig,
  usePetMapping,
  useSaveTransformConfig,
  useTransformCode,
  useTransformComparisons,
  useTransformConfigs,
} from "@/hooks/api/useTransforms";
import { useResetDataset } from "@/hooks/api/useAnalysis";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { calculateKpiSummary } from "@/lib/kpi";
import { downloadBlob, downloadText } from "@/lib/download";
import { KPIBanner } from "@/components/data/KPIBanner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Code2, Download, FolderOpen, Loader2, RotateCcw, Save, Shield, TrendingDown, Wand2, X } from "lucide-react";
import { toast } from "sonner";
import { humanize } from "@/lib/utils";
import type { AppError, PetMappingResponse, TransformDecision, TransformationComparison } from "@/lib/api/types";

const errorText = (e: unknown) => (e as Partial<AppError>)?.message ?? "Please try again.";
const num = (v: number | undefined) => (typeof v === "number" ? v.toFixed(2) : "—");

function petMarkdown(pet: PetMappingResponse) {
  const lines = [
    "# Privacy Protection Mapping",
    "",
    `${pet.summary.protected_columns} of ${pet.summary.risk_columns} at-risk columns protected (${pet.summary.protection_rate.toFixed(1)}%).`,
    "",
    "| Column | Risk | Suggested technique | Status |",
    "|---|---|---|---|",
    ...pet.pet_mapping.map((p) => `| ${p.column} | ${p.risk_level} | ${p.suggested_pet ?? "—"} | ${p.status_text} |`),
  ];
  return lines.join("\n");
}

export function TransformActions() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const decisions = useWorkspaceStore((s) => s.acceptedDecisions);
  const columnCount = useWorkspaceStore((s) => s.columns.length);
  const pendingFeedback = useWorkspaceStore((s) => s.pendingFeedback);
  const { toggleDecision, setDecisions, setFeedback, setKpiSummary } = useWorkspaceStore();

  const comparisons = useTransformComparisons();
  const pet = usePetMapping();
  const apply = useApplyTransforms();
  const code = useTransformCode();
  const save = useSaveTransformConfig();
  const load = useLoadTransformConfig();
  const rollback = useResetDataset();

  const [impact, setImpact] = useState<TransformationComparison[] | null>(null);
  const [petResult, setPetResult] = useState<PetMappingResponse | null>(null);
  const [transformedCsv, setTransformedCsv] = useState<string | null>(null);
  const [codeText, setCodeText] = useState<string | null>(null);
  const [loadOpen, setLoadOpen] = useState(false);
  const [confirmRollback, setConfirmRollback] = useState(false);
  const configs = useTransformConfigs(fileId, loadOpen);

  if (!fileId) return null;
  const input = { fileId, decisions };
  const kpi = impact ? calculateKpiSummary(impact, columnCount) : null;

  function previewImpact() {
    comparisons.mutate(input, {
      onSuccess: (res) => {
        setImpact(res);
        setKpiSummary(calculateKpiSummary(res, columnCount)); // also shown on the Upload page
      },
      onError: (e) => toast.error("Preview failed", { description: errorText(e) }),
    });
  }

  function applyForGeneration() {
    apply.mutate(input, {
      onSuccess: (res) => {
        // Generation applies these to the data before training (feedback items, params at top level)
        setFeedback(decisions.map((d) => ({ column: d.column, action: d.transformation, ...d.params })));
        setTransformedCsv(res.transformed_data?.content ?? null);
        const skipped = res.summary.skipped_actions;
        toast.success(`${res.summary.applied_count} transformation(s) ready for generation`, {
          description: skipped.length ? `Skipped: ${skipped.map((s) => `${s.column} (${s.reason})`).join(", ")}` : undefined,
        });
      },
      onError: (e) => toast.error("Apply failed", { description: errorText(e) }),
    });
  }

  async function downloadTransformed() {
    if (transformedCsv) downloadBlob(await decodeTransformedData(transformedCsv), "transformed_data.csv");
  }

  // mutateAsync, not mutate: a successful rollback clears the accepted decisions, which unmounts
  // this panel, and mutate() skips its callbacks once the caller has unmounted
  async function doRollback() {
    try {
      const res = await rollback.mutateAsync(fileId!);
      toast.success("Transformations rolled back", { description: res.message });
      setImpact(null);
      setTransformedCsv(null);
      setConfirmRollback(false);
    } catch (e) {
      toast.error("Rollback failed", { description: errorText(e) });
    }
  }

  return (
    <div className="rounded-lg border p-3 space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <p className="text-sm font-medium">Accepted ({decisions.length})</p>
        {decisions.map((d: TransformDecision) => (
          <Badge key={`${d.column}|${d.category}|${d.transformation}`} variant="secondary" className="gap-1">
            <span className="font-mono">{d.column}</span> · {humanize(d.transformation)}
            <button aria-label={`Remove ${d.column} ${d.transformation}`} onClick={() => toggleDecision(d)}>
              <X className="h-3 w-3" />
            </button>
          </Badge>
        ))}
      </div>

      <div className="flex flex-wrap gap-2">
        <Button size="sm" variant="outline" className="gap-1" onClick={previewImpact} disabled={comparisons.isPending}>
          {comparisons.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <TrendingDown className="h-3.5 w-3.5" />}
          Preview impact
        </Button>
        <Button size="sm" variant="outline" className="gap-1" disabled={pet.isPending}
          onClick={() => pet.mutate(input, { onSuccess: setPetResult, onError: (e) => toast.error("Mapping failed", { description: errorText(e) }) })}>
          <Shield className="h-3.5 w-3.5" /> Privacy mapping
        </Button>
        <Button size="sm" className="gap-1" onClick={applyForGeneration} disabled={apply.isPending}>
          {apply.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Wand2 className="h-3.5 w-3.5" />}
          Apply for generation
        </Button>
        <Button size="sm" variant="outline" className="gap-1" disabled={code.isPending}
          onClick={() => code.mutate(input, { onSuccess: setCodeText, onError: (e) => toast.error("Code failed", { description: errorText(e) }) })}>
          <Code2 className="h-3.5 w-3.5" /> Show code
        </Button>
        <Button size="sm" variant="outline" className="gap-1" disabled={save.isPending}
          onClick={() => save.mutate(input, {
            onSuccess: (r) => toast.success("Configuration saved", { description: r.config_path }),
            onError: (e) => toast.error("Save failed", { description: errorText(e) }),
          })}>
          <Save className="h-3.5 w-3.5" /> Save config
        </Button>
        <Button size="sm" variant="outline" className="gap-1" onClick={() => setLoadOpen(true)}>
          <FolderOpen className="h-3.5 w-3.5" /> Load config
        </Button>
        {confirmRollback ? (
          <Button size="sm" variant="destructive" className="gap-1" onClick={doRollback} disabled={rollback.isPending}>
            <RotateCcw className="h-3.5 w-3.5" /> Confirm roll back
          </Button>
        ) : (
          <Button size="sm" variant="ghost" className="gap-1" onClick={() => setConfirmRollback(true)}>
            <RotateCcw className="h-3.5 w-3.5" /> Roll back
          </Button>
        )}
      </div>

      {pendingFeedback.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 text-sm text-green-700">
          {pendingFeedback.length} transformation(s) will be applied when you generate synthetic data.
          {transformedCsv && (
            <Button size="sm" variant="link" className="h-auto p-0 gap-1" onClick={downloadTransformed}>
              <Download className="h-3.5 w-3.5" /> Download transformed CSV
            </Button>
          )}
        </div>
      )}

      {kpi && impact && (
        <div className="space-y-2">
          <KPIBanner summary={kpi} />
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Column</TableHead>
                <TableHead>Transformation</TableHead>
                <TableHead>Skewness</TableHead>
                <TableHead>Outliers</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {impact.map((c, i) => (
                <TableRow key={i}>
                  <TableCell className="font-mono text-xs">{c.column}</TableCell>
                  <TableCell>{humanize(c.action ?? "")}</TableCell>
                  <TableCell className="text-xs">
                    {c.metrics.is_numeric ? `${num(c.metrics.original.skewness)} → ${num(c.metrics.transformed.skewness)}` : "n/a (text)"}
                  </TableCell>
                  <TableCell className="text-xs">
                    {c.metrics.is_numeric
                      ? `${num(c.metrics.original.outlier_percentage)}% → ${num(c.metrics.transformed.outlier_percentage)}%`
                      : "n/a (text)"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      {petResult && (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-sm">
              <span className="font-medium">Privacy protection:</span> {petResult.summary.protected_columns} of{" "}
              {petResult.summary.risk_columns} at-risk columns protected ({petResult.summary.protection_rate.toFixed(0)}%)
            </p>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={() => downloadText(JSON.stringify(petResult, null, 2), "privacy_mapping.json", "application/json")}>
                Export JSON
              </Button>
              <Button size="sm" variant="outline" onClick={() => downloadText(petMarkdown(petResult), "privacy_mapping.md", "text/markdown")}>
                Export Markdown
              </Button>
            </div>
          </div>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Column</TableHead>
                <TableHead>Risk</TableHead>
                <TableHead>Suggested technique</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {petResult.pet_mapping.map((p) => (
                <TableRow key={p.column}>
                  <TableCell className="font-mono text-xs">{p.column}</TableCell>
                  <TableCell className="text-xs">{p.risk_level}</TableCell>
                  <TableCell className="text-xs">{p.suggested_pet ? humanize(p.suggested_pet) : "—"}</TableCell>
                  <TableCell className={p.is_protected ? "text-xs text-green-600" : "text-xs text-muted-foreground"}>{p.status_text}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      <Dialog open={codeText !== null} onOpenChange={(o) => !o && setCodeText(null)}>
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>Transformation code</DialogTitle>
          </DialogHeader>
          <pre className="max-h-96 overflow-auto rounded bg-muted p-3 text-xs">{codeText}</pre>
          <div className="flex justify-end gap-2">
            <Button size="sm" variant="outline" onClick={() => { navigator.clipboard.writeText(codeText ?? ""); toast.success("Copied"); }}>
              Copy
            </Button>
            <Button size="sm" variant="outline" onClick={() => downloadText(codeText ?? "", "transformations.py", "text/x-python")}>
              Download .py
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={loadOpen} onOpenChange={setLoadOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Load a saved configuration</DialogTitle>
          </DialogHeader>
          {configs.isLoading && <Loader2 className="h-4 w-4 animate-spin" />}
          {configs.data?.length === 0 && <p className="text-sm text-muted-foreground">No saved configurations for this dataset.</p>}
          <div className="space-y-2">
            {configs.data?.map((c) => (
              <div key={c.filepath} className="flex items-center justify-between gap-2 rounded border px-3 py-2 text-sm">
                <span>
                  {new Date(c.created_at).toLocaleString()} · {c.decision_count} decision(s)
                </span>
                <Button size="sm" variant="outline" disabled={load.isPending}
                  onClick={() => load.mutate({ fileId, path: c.filepath }, {
                    onSuccess: (list) => { setDecisions(list); setLoadOpen(false); toast.success(`Loaded ${list.length} decision(s)`); },
                    onError: (e) => toast.error("Load failed", { description: errorText(e) }),
                  })}>
                  Load
                </Button>
              </div>
            ))}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
