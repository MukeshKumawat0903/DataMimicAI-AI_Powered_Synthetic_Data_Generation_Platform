"use client";

import { useGenerate } from "@/hooks/api/useGenerate";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { downloadText } from "@/lib/download";
import { Button } from "@/components/ui/button";
import { CheckCircle2, FileDown, Wand2 } from "lucide-react";
import { toast } from "sonner";
import type { Algorithm, AppError, GeneratorConfig, QualityReportResponse } from "@/lib/api/types";

// Models whose training length is set by epochs
const EPOCH_MODELS = new Set(["CTGAN", "TVAE", "PARS", "ddpm", "ctgan", "tvae", "dpgan", "pategan"]);

interface Recommendation {
  text: string;
  label?: string;
  apply?: { algorithm?: Algorithm; epochs?: number; num_rows?: number };
}

function buildRecommendations(data: QualityReportResponse, algorithm: Algorithm, config: GeneratorConfig): Recommendation[] {
  const recs: Recommendation[] = [];
  const epochs = config.epochs ?? 300;
  const rows = config.num_rows;
  const { fidelity, privacy, utility } = data.scores;
  const copies = data.privacy_detail ? 1 - data.privacy_detail.new_row_synthesis : 0;

  if (fidelity < 75) {
    recs.push(
      EPOCH_MODELS.has(algorithm)
        ? { text: `Fidelity is ${fidelity}. Training longer lets the model learn the distributions better.`,
            label: `Train longer (${epochs} → ${Math.min(2000, epochs * 2)} epochs)`, apply: { epochs: Math.min(2000, epochs * 2) } }
        : { text: `Fidelity is ${fidelity}. ${algorithm} assumes simple shapes; CTGAN captures complex distributions better.`,
            label: "Switch to CTGAN", apply: { algorithm: "CTGAN", epochs: 300 } }
    );
  }
  if ((privacy !== null && privacy < 60) || copies > 0.05) {
    recs.push(
      EPOCH_MODELS.has(algorithm) && epochs > 20
        ? { text: "Synthetic rows sit close to real records. Fewer epochs reduce memorisation.",
            label: `Train less (${epochs} → ${Math.max(10, Math.round(epochs / 2))} epochs)`, apply: { epochs: Math.max(10, Math.round(epochs / 2)) } }
        : { text: "Synthetic rows sit close to real records. GaussianCopula is less prone to copying rows.",
            label: "Switch to GaussianCopula", apply: { algorithm: "GaussianCopula" } }
    );
  }
  if (utility < 60) {
    recs.push({ text: `Utility is ${utility}: category mixes differ. More rows give a closer match.`,
      label: `Generate more rows (${rows} → ${rows * 2})`, apply: { num_rows: rows * 2 } });
  }
  if (recs.length === 0) recs.push({ text: "Quality looks good. Download the data, or try another model to compare." });
  return recs;
}

function reportMarkdown(data: QualityReportResponse, algorithm: string) {
  const s = data.scores;
  const lines = [
    "# Synthetic data quality report",
    "",
    `Model: ${algorithm} · ${new Date().toLocaleString()}`,
    "",
    `**Scores:** fidelity ${s.fidelity} · privacy ${s.privacy ?? "n/a"} · utility ${s.utility}`,
  ];
  if (data.privacy_detail)
    lines.push("", `Privacy: DCR baseline protection ${data.privacy_detail.dcr_baseline_protection.toFixed(2)}; ${Math.round(data.privacy_detail.new_row_synthesis * 100)}% of synthetic rows are new.`);
  if (data.summary) {
    const r = data.summary.real, y = data.summary.synthetic;
    lines.push("", "## Key metrics", "", "| Metric | Real | Synthetic |", "|---|---|---|",
      `| Rows | ${r.rows} | ${y.rows} |`, `| Columns | ${r.columns} | ${y.columns} |`,
      `| Missing values | ${r.missing_pct}% | ${y.missing_pct}% |`, `| Duplicate rows | ${r.duplicate_rows} | ${y.duplicate_rows} |`);
  }
  lines.push("", "## Issues", "", ...(data.issues.length ? data.issues.map((i) => `- [${i.severity}] ${i.message}`) : ["- None"]));
  lines.push("", "## Per-column metrics", "", "| Column | Type | KS | p-value |", "|---|---|---|---|",
    ...data.column_metrics.map((c) => `| ${c.name} | ${c.type} | ${c.ks_stat?.toFixed(3) ?? "—"} | ${c.p_value?.toFixed(4) ?? "—"} |`));
  return lines.join("\n");
}

function metricsCsv(data: QualityReportResponse) {
  return ["column,type,ks_stat,p_value", ...data.column_metrics.map((c) => `${c.name},${c.type},${c.ks_stat ?? ""},${c.p_value ?? ""}`)].join("\n");
}

export function QualityActions({ data }: { data: QualityReportResponse }) {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const { selectedAlgorithm, generatorConfig, pendingFeedback, setSelectedAlgorithm, setGeneratorConfig } = useWorkspaceStore();
  const generate = useGenerate();
  const recs = buildRecommendations(data, selectedAlgorithm, generatorConfig);

  function apply(rec: Recommendation) {
    if (!fileId || !rec.apply) return;
    const algorithm = rec.apply.algorithm ?? selectedAlgorithm;
    const config = {
      ...generatorConfig,
      ...(rec.apply.epochs ? { epochs: rec.apply.epochs } : {}),
      ...(rec.apply.num_rows ? { num_rows: rec.apply.num_rows } : {}),
    };
    if (rec.apply.algorithm) setSelectedAlgorithm(algorithm);
    setGeneratorConfig(config);
    toast.info(`Regenerating: ${rec.label}`);
    generate.mutate(
      { file_id: fileId, feedback: pendingFeedback, generator_config: { ...config, algorithm } },
      {
        onSuccess: () => toast.success("New version generated", { description: "Scores refresh automatically." }),
        onError: (e) => toast.error("Regeneration failed", { description: (e as Partial<AppError>).message }),
      }
    );
  }

  return (
    <div className="space-y-3">
      <div className="rounded-lg border border-blue-200 bg-blue-50 dark:bg-blue-950/20 dark:border-blue-800 p-3 space-y-2">
        <p className="text-xs font-medium text-blue-700 dark:text-blue-400">Recommendations</p>
        {recs.map((r) => (
          <div key={r.text} className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-sm text-blue-700 dark:text-blue-300 flex items-start gap-2">
              <CheckCircle2 className="h-4 w-4 shrink-0 mt-0.5" />
              {r.text}
            </p>
            {r.apply && (
              <Button size="sm" variant="outline" className="gap-1 bg-background" disabled={generate.isPending} onClick={() => apply(r)}>
                <Wand2 className="h-3.5 w-3.5" /> {generate.isPending ? "Generating…" : r.label}
              </Button>
            )}
          </div>
        ))}
      </div>
      <div className="flex flex-wrap gap-2">
        <Button size="sm" variant="outline" className="gap-1"
          onClick={() => downloadText(reportMarkdown(data, selectedAlgorithm), "quality_report.md", "text/markdown")}>
          <FileDown className="h-3.5 w-3.5" /> Download report
        </Button>
        <Button size="sm" variant="outline" className="gap-1" onClick={() => downloadText(metricsCsv(data), "quality_metrics.csv", "text/csv")}>
          <FileDown className="h-3.5 w-3.5" /> Export metrics (CSV)
        </Button>
      </div>
    </div>
  );
}
