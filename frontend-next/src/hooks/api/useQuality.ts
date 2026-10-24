import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { QualityIssue, QualityReportResponse, QualityScore, TableSummary } from "@/lib/api/types";

interface ColumnMetric {
  type: "numeric" | "categorical";
  ks_stat?: number;
  p_value?: number;
  mean_real?: number;
  mean_synth?: number;
  chi2?: number;
  error?: string;
}

interface MetricsResponse {
  file_id: string;
  metrics: Record<string, ColumnMetric>;
  privacy?: { dcr_baseline_protection?: number; new_row_synthesis?: number; error?: string };
  summary?: { real: TableSummary; synthetic: TableSummary };
}

function deriveQualityReport(res: MetricsResponse): QualityReportResponse {
  const entries = Object.values(res.metrics).filter((m) => !m.error);
  const numericEntries = entries.filter((m) => m.type === "numeric" && m.ks_stat !== undefined);
  const categoricalEntries = entries.filter((m) => m.type === "categorical" && m.p_value !== undefined);

  // Fidelity: based on KS statistics (lower KS = better)
  const avgKS = numericEntries.length > 0
    ? numericEntries.reduce((acc, m) => acc + (m.ks_stat ?? 0), 0) / numericEntries.length
    : 0;
  const fidelity = Math.max(0, Math.min(100, Math.round((1 - avgKS) * 100)));

  // Utility: based on p-values (higher p = better distribution match)
  const avgP = categoricalEntries.length > 0
    ? categoricalEntries.reduce((acc, m) => acc + (m.p_value ?? 0), 0) / categoricalEntries.length
    : 0.5;
  const utility = Math.round(Math.min(100, avgP * 100 + 30));

  // Privacy: distance from synthetic rows to their closest real record, vs. random data (sdmetrics)
  const dcr = res.privacy?.dcr_baseline_protection;
  const newRows = res.privacy?.new_row_synthesis;
  const privacy = dcr !== undefined ? Math.round(dcr * 100) : null;

  const issues: QualityIssue[] = [];
  if (fidelity < 60) {
    issues.push({
      category: "fidelity",
      severity: "high",
      message: "Low distribution fidelity — synthetic data may not match the original well.",
    });
  }
  if (avgKS > 0.3) {
    issues.push({
      category: "fidelity",
      severity: "medium",
      message: `Average KS statistic is ${avgKS.toFixed(2)} — consider more training epochs.`,
    });
  }
  if (newRows !== undefined && newRows < 0.95) {
    issues.push({
      category: "privacy",
      severity: newRows < 0.8 ? "high" : "medium",
      message: `${Math.round((1 - newRows) * 100)}% of synthetic rows copy a real row.`,
    });
  }
  // Column-level differences (KS > 0.3: clearly different distribution; p < 0.05: different category mix)
  for (const [col, m] of Object.entries(res.metrics)) {
    if (m.type === "numeric" && (m.ks_stat ?? 0) > 0.3) {
      issues.push({ category: "fidelity", severity: "medium", affected_column: col,
        message: `${col}: distribution differs from the real data (KS ${m.ks_stat!.toFixed(2)}).` });
    }
    if (m.type === "categorical" && m.p_value !== undefined && m.p_value < 0.05) {
      issues.push({ category: "utility", severity: "low", affected_column: col,
        message: `${col}: category mix differs from the real data (p ${m.p_value.toFixed(3)}).` });
    }
  }

  const scores: QualityScore = { fidelity, privacy, utility };
  const weakest = (Object.entries(scores) as [keyof QualityScore, number | null][])
    .filter((entry): entry is [keyof QualityScore, number] => entry[1] !== null)
    .sort((a, b) => a[1] - b[1])[0][0];

  return {
    scores,
    issues,
    recommendation: weakest === "fidelity"
      ? "Increase training epochs or switch to CTGAN/DDPM for better fidelity."
      : weakest === "privacy"
      ? "Synthetic rows sit close to real records. Try a privacy-focused model (PrivBayes, DP-GAN, PATE-GAN) or fewer epochs."
      : "Add more real training data or reduce num_rows for better utility.",
    weakest_category: weakest,
    summary: res.summary,
    column_metrics: Object.entries(res.metrics).map(([name, m]) => ({
      name,
      type: m.type,
      ks_stat: m.ks_stat,
      p_value: m.p_value,
    })),
    privacy_detail: dcr !== undefined && newRows !== undefined
      ? { dcr_baseline_protection: dcr, new_row_synthesis: newRows }
      : undefined,
  };
}

export function useQuality(fileId: string | null) {
  return useQuery<QualityReportResponse>({
    queryKey: ["quality", fileId],
    queryFn: async () => {
      const res = await apiClient.get<MetricsResponse>(`/metrics`, {
        params: { file_id: fileId },
      });
      return deriveQualityReport(res.data);
    },
    enabled: !!fileId,
    staleTime: 2 * 60 * 1000,
  });
}
