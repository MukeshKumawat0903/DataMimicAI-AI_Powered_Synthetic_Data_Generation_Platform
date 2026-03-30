/**
 * API type definitions for DataMimicAI backend (FastAPI).
 * These types correspond to the backend response shapes and request payloads.
 */

// ─── Common ────────────────────────────────────────────────────────────────

export interface AppError {
  status: number;
  message: string;
  detail?: string;
}

// ─── Upload ────────────────────────────────────────────────────────────────

export interface UploadResponse {
  file_id: string;
  columns: string[];
  num_rows: number;
  num_columns: number;
  sample_data: Record<string, unknown>[];
}

export interface LoadDemoResponse {
  file_id: string;
  algorithm: string;
  columns: string[];
  num_rows: number;
  sample_data: Record<string, unknown>[];
}

// ─── EDA ───────────────────────────────────────────────────────────────────

export type ColumnType = "numeric" | "categorical" | "datetime" | "text" | "unknown";

export interface ColumnInfo {
  name: string;
  type: ColumnType;
  missing_pct: number;
  unique_count: number;
  is_pii?: boolean;
}

export interface DatasetResponse {
  file_id: string;
  columns: ColumnInfo[];
  num_rows: number;
  num_columns: number;
  sample_data: Record<string, unknown>[];
}

// ─── KPI / Smart Preview ───────────────────────────────────────────────────

export interface TransformationComparison {
  column: string;
  metrics: {
    is_numeric: boolean;
    original: {
      skewness?: number;
      outlier_percentage?: number;
      error?: string;
    };
    transformed: {
      skewness?: number;
      outlier_percentage?: number;
      error?: string;
    };
    deltas: {
      skewness?: number | { absolute: number };
      outlier_percentage?: number | { absolute: number };
    };
  };
}

export interface KPISummary {
  pct_transformed: number;
  avg_skewness_change: number;
  avg_outlier_change: number;
  risky_columns: Array<{
    column: string;
    reasons: string[];
    skewness?: number;
    outlier_pct?: number;
  }>;
  improved_count: number;
  worsened_count: number;
  total_transformed: number;
}

// ─── EDA: Profiling ────────────────────────────────────────────────────────

export interface ProfileColumn {
  column: string;
  dtype: string;
  missing_pct: number;
  unique_count: number;
  top_value?: string | number;
  suggestions: string[];
  is_pii?: boolean;
}

export interface ProfileResponse {
  file_id: string;
  columns: ProfileColumn[];
  row_count: number;
}

// ─── EDA: Correlation ──────────────────────────────────────────────────────

export interface CorrelationPair {
  col_a: string;
  col_b: string;
  correlation: number;
  is_leakage: boolean;
}

export interface CorrelationResponse {
  heatmap_base64: string;
  top_pairs: CorrelationPair[];
}

// ─── EDA: Outliers ─────────────────────────────────────────────────────────

export interface OutlierResult {
  column: string;
  method: string;
  outlier_count: number;
  pct_affected: number;
}

export interface OutliersResponse {
  results: OutlierResult[];
  methods_used: string[];
  file_id: string;
}

export interface RemediateRequest {
  file_id: string;
  remediations: Array<{
    column: string;
    action: "remove" | "cap" | "transform" | "impute";
  }>;
}

// ─── EDA: Privacy ──────────────────────────────────────────────────────────

export interface PIIResult {
  column: string;
  pii_type: string;
  confidence: number;
  sample_masked: string;
}

export interface KAnonymityResult {
  k_value: number;
  threshold: number;
  quasi_identifiers: string[];
  at_risk_records: number;
}

export interface PrivacyResponse {
  pii_results: PIIResult[];
  k_anonymity: KAnonymityResult;
  scan_mode: "fast" | "deep";
}

// ─── EDA: Feature Suggestions ─────────────────────────────────────────────

export interface FeatureSuggestion {
  column: string;
  suggestion_type: string;
  description: string;
  impact: "high" | "medium" | "low";
  resolved: boolean;
}

export interface FeaturesResponse {
  file_id?: string;
  /** Utility-focused suggestions from /eda/context-aware-suggestions */
  utility_suggestions: FeatureSuggestion[];
  /** Privacy-focused suggestions */
  privacy_suggestions: FeatureSuggestion[];
  conflicts: Array<{ column?: string; col_a?: string; col_b?: string; utility?: string; privacy?: string; reason?: string; resolution?: string }>;
  conflict_summary?: Record<string, unknown>;
  non_conflicting_utility?: FeatureSuggestion[];
  non_conflicting_privacy?: FeatureSuggestion[];
  metadata?: Record<string, unknown>;
}

// ─── EDA: Time Series ─────────────────────────────────────────────────────

export interface TimeSeriesMetric {
  column: string;
  is_stationary: boolean;
  acf_values: number[];
  pacf_values: number[];
  trend: "increasing" | "decreasing" | "stable" | "unknown";
}

export interface TimeSeriesResponse {
  detected_columns: string[];
  metrics: TimeSeriesMetric[];
}

// ─── LLM / Diagnostics ─────────────────────────────────────────────────────

export interface LLMExplanationStep {
  step: number;
  label: string;
  status: "pending" | "running" | "done" | "error";
  detail?: string;
}

export interface LLMExplanationResponse {
  explanation: string;
  validation_pct: number;
  steps: LLMExplanationStep[];
}

// ─── Generation ────────────────────────────────────────────────────────────

export type SDVAlgorithm = "CTGAN" | "GaussianCopula" | "TVAE" | "PARS";
export type SynthCityAlgorithm = "ddpm" | "ctgan" | "tvae" | "privbayes" | "dpgan" | "pategan" | "arf";
export type Algorithm = SDVAlgorithm | SynthCityAlgorithm;

export interface GeneratorConfig {
  algorithm: Algorithm;
  num_rows: number;
  epochs?: number;
  num_sequences?: number;
  sequence_length?: number;
  context_columns?: string[];
}

export interface FeedbackItem {
  column: string;
  action: string;
  params?: Record<string, unknown>;
}

export interface GenerationRequest {
  file_id: string;
  feedback: FeedbackItem[];
  generator_config: GeneratorConfig;
}

export interface GenerationResponse {
  // The endpoint streams CSV — this is the meta header from Content-Disposition
  synthetic_file_id?: string;
  rows_generated?: number;
}

// ─── Validation ────────────────────────────────────────────────────────────

export interface QualityScore {
  fidelity: number;
  privacy: number;
  utility: number;
}

export interface QualityIssue {
  category: "fidelity" | "privacy" | "utility";
  severity: "high" | "medium" | "low";
  message: string;
  affected_column?: string;
}

export interface QualityReportResponse {
  scores: QualityScore;
  issues: QualityIssue[];
  recommendation: string;
  weakest_category: keyof QualityScore;
}

export interface ValidationMetric {
  metric: string;
  column: string;
  before: number;
  after: number;
  delta: number;
  status: "SUCCESS" | "FAILED" | "IMPROVED" | "DEGRADED";
  error?: string | null;
}

export interface DecisionReportResponse {
  plan_id: string;
  validation_results: ValidationMetric[];
  summary: {
    metrics_compared: number;
    columns_affected: number;
    improved: number;
    degraded: number;
  };
}

// ─── Planner / Approval / Execution ────────────────────────────────────────

export interface DiagnosticsInterpretation {
  issues_found: number;
  severity: "critical" | "warning" | "info";
  summary: string;
}

export interface PlanStep {
  id: string;
  action: string;
  column: string;
  reason: string;
  impact: "high" | "medium" | "low";
  approved: boolean | null;
}

export interface Plan {
  plan_id: string;
  steps: PlanStep[];
  created_at: string;
}

export interface ExecutionStatus {
  plan_id: string;
  status: "pending" | "running" | "complete" | "failed";
  progress: number;
  message: string;
}
