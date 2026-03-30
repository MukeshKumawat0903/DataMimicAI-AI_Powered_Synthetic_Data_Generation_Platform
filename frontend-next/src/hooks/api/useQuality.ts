import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { QualityReportResponse } from "@/lib/api/types";

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
}

function deriveQualityReport(metrics: Record<string, ColumnMetric>): QualityReportResponse {
  const entries = Object.values(metrics).filter((m) => !m.error);
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

  // Privacy: placeholder (no direct privacy metric from /metrics endpoint)
  const privacy = 75;

  const issues = [];
  if (fidelity < 60) {
    issues.push({
      category: "fidelity" as const,
      severity: "high" as const,
      message: "Low distribution fidelity — synthetic data may not match the original well.",
    });
  }
  if (avgKS > 0.3) {
    issues.push({
      category: "fidelity" as const,
      severity: "medium" as const,
      message: `Average KS statistic is ${avgKS.toFixed(2)} — consider more training epochs.`,
    });
  }

  const scores = { fidelity, privacy, utility };
  const weakest = (Object.entries(scores) as [keyof typeof scores, number][])
    .sort((a, b) => a[1] - b[1])[0][0];

  return {
    scores,
    issues,
    recommendation: weakest === "fidelity"
      ? "Increase training epochs or switch to CTGAN/DDPM for better fidelity."
      : weakest === "privacy"
      ? "Run the Privacy Audit to detect quasi-identifiers before sharing data."
      : "Add more real training data or reduce num_rows for better utility.",
    weakest_category: weakest,
  };
}

export function useQuality(fileId: string | null) {
  return useQuery<QualityReportResponse>({
    queryKey: ["quality", fileId],
    queryFn: async () => {
      const res = await apiClient.get<MetricsResponse>(`/metrics`, {
        params: { file_id: fileId },
      });
      return deriveQualityReport(res.data.metrics);
    },
    enabled: !!fileId,
    staleTime: 2 * 60 * 1000,
  });
}
