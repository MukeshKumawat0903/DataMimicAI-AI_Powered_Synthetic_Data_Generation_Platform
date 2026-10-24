import type { KPISummary, TransformationComparison } from "@/lib/api/types";

const deltaValue = (d: number | { absolute: number } | undefined) =>
  d === undefined || d === null ? undefined : typeof d === "number" ? d : d.absolute;

/**
 * Transformation Impact Summary from before/after comparisons
 * (same thresholds as the Streamlit Smart Preview's calculate_kpi_summary).
 */
export function calculateKpiSummary(comparisons: TransformationComparison[], totalColumns: number): KPISummary {
  const transformed = new Set<string>();
  const skewChanges: number[] = [];
  const outlierChanges: number[] = [];
  const risky: KPISummary["risky_columns"] = [];
  let improved = 0;
  let worsened = 0;

  for (const comp of comparisons) {
    if (comp.column) transformed.add(comp.column);
    const m = comp.metrics;
    if (!m?.is_numeric || m.original?.error || m.transformed?.error) continue;

    const skew = deltaValue(m.deltas?.skewness);
    const outliers = deltaValue(m.deltas?.outlier_percentage);
    if (skew !== undefined) skewChanges.push(skew);
    if (outliers !== undefined) outlierChanges.push(outliers);

    // Improved if skewness drops by > 0.1 or outliers by > 1 percentage point
    const isImproved = (skew !== undefined && skew < -0.1) || (outliers !== undefined && outliers < -1);
    const isWorse = (skew !== undefined && skew > 0.1) || (outliers !== undefined && outliers > 1);
    if (isImproved) improved++;
    else if (isWorse) worsened++;

    // Still risky after transformation
    const reasons: string[] = [];
    const tSkew = m.transformed?.skewness;
    const tOut = m.transformed?.outlier_percentage;
    if (typeof tSkew === "number" && Math.abs(tSkew) > 2) reasons.push(`High skewness: ${tSkew.toFixed(2)}`);
    if (typeof tOut === "number" && tOut > 10) reasons.push(`High outliers: ${tOut.toFixed(1)}%`);
    if (reasons.length) risky.push({ column: comp.column, reasons, skewness: tSkew, outlier_pct: tOut });
  }

  const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
  return {
    pct_transformed: totalColumns ? (transformed.size / totalColumns) * 100 : 0,
    avg_skewness_change: avg(skewChanges),
    avg_outlier_change: avg(outlierChanges),
    risky_columns: risky,
    improved_count: improved,
    worsened_count: worsened,
    total_transformed: transformed.size,
  };
}
