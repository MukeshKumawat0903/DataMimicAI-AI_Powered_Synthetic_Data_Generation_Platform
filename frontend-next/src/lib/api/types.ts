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
  action?: string;
  transformed_column?: string;
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
  /** Fixable suggestions (fix_type: impute, encode, outlier, drop_*; "review" has no automatic fix) */
  fixes: { action: string; fix_type: string }[];
  is_pii?: boolean;
}

export interface ProfileResponse {
  file_id: string;
  columns: ProfileColumn[];
  row_count: number;
  summary: {
    n_rows: number;
    n_columns: number;
    total_missing: number;
    columns_with_missing: number;
    constant_columns: string[];
    duplicated_columns: string[];
    columns_high_missing: string[];
  };
}

/** POST /eda/compute-distributions/{id} */
export interface DistributionsResponse {
  distributions: Record<string, {
    best_distribution: string;
    score: number | null;
    skewness: number | null;
    kurtosis: number | null;
    error?: string | null;
  }>;
  summary: {
    total_columns_analyzed: number;
    distribution_counts: Record<string, number>;
    highly_skewed_columns: string[];
    heavy_tailed_columns: string[];
    recommendations: string[];
  };
}

/** POST /eda/compute-mi-matrix/{id} */
export interface MutualInfoResponse {
  top_mi_pairs: { column1: string; column2: string; mi_score: number }[];
  nonlinear_analysis: {
    nonlinear_relationships: { column1: string; column2: string; mi_score: number; pearson_score: number }[];
    recommendations: string[];
  };
}

/** GET /eda/detect-dependencies/{id} */
export interface DependenciesResponse {
  dependencies: { determinant: string; dependent: string; confidence: number; type: string }[];
  total_found: number;
  graph_data: { summary: { total_dependencies: number; columns_involved: number; strict_dependencies: number } };
}

/** POST /eda/outliers/detect-cooks-distance */
export interface CooksDistanceResponse {
  stats: {
    target: string;
    features: string[];
    outlier_count: number;
    outlier_percent: number;
    threshold: number;
    max_cooks_d: number;
  }[];
}

/** GET /eda/suggest-quasi-identifiers/{id} */
export interface QuasiIdentifier {
  column: string;
  unique_values: number;
  uniqueness_ratio: number;
  risk_level: string;
  sample_values: unknown[];
}

/** POST /eda/compute-k-anonymity */
export interface KAnonymityResponse {
  risk_report: {
    summary: { severity: string; unique_records: number; average_k: number };
    statistics: { min_k: number; max_k: number; median_k: number; unique_percentage: number };
    recommendations: { priority: string; issue: string; recommendation: string }[];
  };
  vulnerable_count: number;
  k_threshold: number;
  total_records: number;
}

interface Correlogram {
  values: number[];
  lower: number[];
  upper: number[];
  lags: number[];
}

/** POST /eda/timeseries-acf-pacf */
export interface AcfPacfResponse {
  acf_pacf: {
    acf: Correlogram;
    pacf: Correlogram;
    suggestions: { model_type: string; explanation: string };
    n_obs: number;
  };
}

/** POST /eda/timeseries-decompose */
export interface DecompositionResponse {
  decomposition: {
    observed: { values: number[]; index: string[] };
    trend: { values: (number | null)[]; index: string[] };
    seasonal: { values: number[]; index: string[] };
    residual: { values: (number | null)[]; index: string[] };
    model: string;
    period: number;
    strength_of_trend: number;
    strength_of_seasonality: number;
  };
}

// ─── EDA: Correlation ──────────────────────────────────────────────────────

export interface CorrelationPair {
  col_a: string;
  col_b: string;
  correlation: number;
  is_leakage: boolean;
}

export type CorrelationMethod = "pearson" | "spearman" | "kendall";

export interface CorrelationResponse {
  heatmap_base64: string;
  top_pairs: CorrelationPair[];
  /** Full numeric correlation matrices, column → column → coefficient */
  matrices: Record<CorrelationMethod, Record<string, Record<string, number>>>;
  /** Cramér's V between categorical columns, strongest first */
  categorical: { col_a: string; col_b: string; cramers_v: number }[];
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
  /** Outlier row indices per column, merged across the detection methods */
  indices: Record<string, number[]>;
}

export type RemediationAction = "cap" | "winsorize" | "remove" | "bin" | "transform";

export interface RemediateRequest {
  file_id: string;
  remediations: Array<{
    column: string;
    action: RemediationAction;
  }>;
  /** From detection; "remove" needs to know which rows to drop */
  indices?: Record<string, number[]>;
}

// ─── EDA: Privacy ──────────────────────────────────────────────────────────

export interface PIIResult {
  column: string;
  pii_type: string;
  confidence: number;
  sample_masked: string;
}

export interface PrivacyResponse {
  pii_results: PIIResult[];
  scan_mode: "fast" | "deep";
  /** Explains why the scan that ran differs from the one requested */
  notice?: string;
}

// ─── EDA: Feature Suggestions ─────────────────────────────────────────────

export interface FeatureSuggestion {
  column: string;
  suggestion_type: string;
  description: string;
  impact: "high" | "medium" | "low";
  resolved: boolean;
  category: "utility" | "privacy";
  params: Record<string, unknown>;
  metrics?: Record<string, unknown>;
}

/** A suggestion the user accepted; the shape the transform endpoints expect */
export interface TransformDecision {
  column: string;
  transformation: string;
  category: "utility" | "privacy";
  params: Record<string, unknown>;
  reason: string;
  metadata?: Record<string, unknown>;
}

export interface FeatureConflict {
  column: string;
  severity?: string;
  reason?: string;
  utility?: FeatureSuggestion;
  privacy?: FeatureSuggestion;
}

export interface FeaturesResponse {
  file_id?: string;
  /** Utility-focused suggestions from /eda/context-aware-suggestions */
  utility_suggestions: FeatureSuggestion[];
  /** Privacy-focused suggestions */
  privacy_suggestions: FeatureSuggestion[];
  conflicts: FeatureConflict[];
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

export interface LLMExplanationResponse {
  explanation: string;
  validated: boolean;
  /** Share of the backend's output checks that passed (0–100) */
  validation_pct: number;
  checks: { label: string; passed: boolean }[];
  /** "facts_summary" when the LLM answer failed the checks and the backend summarised the statistics instead */
  source: "llm" | "facts_summary";
}

/** GET /llm/scopes */
export interface LLMScopes {
  scopes: Record<string, { description: string; requires_columns: boolean }>;
  tones: Record<string, string>;
}

export interface LLMExplanationOptions {
  scope: string;
  tone: string;
  columns?: string[];
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

/** Applied by the backend before generation; transformation params sit at the top level */
export interface FeedbackItem {
  column: string;
  action: string;
  [param: string]: unknown;
}

/** POST /eda/get-pet-mapping: privacy-enhancing technique per column */
export interface PetMappingItem {
  column: string;
  risk_level: string;
  pii_types: string[];
  is_quasi_identifier: boolean;
  confidence: number;
  suggested_pet: string | null;
  suggestion_reason: string | null;
  applied_pet: string | null;
  is_protected: boolean;
  protection_status: string;
  status_text: string;
}

export interface PetMappingResponse {
  pet_mapping: PetMappingItem[];
  summary: {
    total_columns: number;
    risk_columns: number;
    protected_columns: number;
    unprotected_columns: number;
    unprotected_high_risk: number;
    protection_rate: number;
  };
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
  /** null when the backend couldn't compute privacy metrics */
  privacy: number | null;
  utility: number;
}

export interface TableSummary {
  rows: number;
  columns: number;
  missing_pct: number;
  duplicate_rows: number;
  numeric_columns: number;
  categorical_columns: number;
}

/** One synthetic dataset produced in this session */
export interface GenerationRecord {
  version: number;
  algorithm: string;
  numRows: number;
  epochs?: number;
  timestamp: string; // ISO
  scores?: QualityScore;
  /** Backend id of this generation's saved copy (for restore / download) */
  versionId?: string;
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
  /** Real vs synthetic shape/quality facts for the Key Metrics table */
  summary?: { real: TableSummary; synthetic: TableSummary };
  /** Raw privacy metrics behind the privacy score (0–1) */
  privacy_detail?: { dcr_baseline_protection: number; new_row_synthesis: number };
  /** Per-column similarity tests (KS for numeric, chi-square for categorical) */
  column_metrics: { name: string; type: "numeric" | "categorical"; ks_stat?: number; p_value?: number }[];
}

// ─── AI Assistance: diagnostics → plan → approval → execution ─────────────

export interface DiagnosticIssue {
  issue_type: string;
  affected_columns: string[];
  metric_name: string;
  metric_value: number;
  severity: "high" | "medium" | "low";
}

/** POST /api/diagnostics/build */
export interface DiagnosticsBuildResponse {
  diagnostics: {
    diagnostics: DiagnosticIssue[];
    summary: {
      total_issues: number;
      high_severity_count: number;
      medium_severity_count: number;
      low_severity_count: number;
      issue_types: Record<string, number>;
    };
    metadata?: Record<string, unknown>;
  };
  /** Raw dataset signals; passed on to the planner */
  signals: Record<string, unknown>;
}

/** POST /api/diagnostics/interpret → interpretation */
export interface DiagnosticsInterpretation {
  overall_assessment: string;
  dominant_issue_patterns: string[];
  supporting_evidence: {
    pattern: string;
    linked_issue_types: string[];
    affected_columns: string[];
    severity?: string;
  }[];
  confidence: string;
}

/** POST /api/planner/create-plan → proposed_plans[] */
export interface TransformationPlan {
  plan_id: string;
  applicable_issue_patterns: string[];
  proposed_transformations: { transformation: string; target_columns: string[]; rationale: string }[];
  assumptions: string[];
  risks: string[];
}

export type PlanDecision = "APPROVED" | "REJECTED";

/** POST /api/approval/review-plan */
export interface PlanReview {
  plan_id: string;
  status: PlanDecision;
  reviewed_by: string;
  reviewed_at: string;
  notes?: string | null;
}

/** POST /api/execution/execute-plan */
export interface PlanExecution {
  plan_id: string;
  execution_status: string;
  applied_transformations: string[];
  validation_available: boolean;
  validation_plan_id?: string | null;
  error?: string | null;
  validation_error?: string | null;
}

export interface DecisionMetric {
  metric: string;
  column: string;
  before: number | null;
  after: number | null;
  delta: number | null;
  status: string;
  error?: string | null;
}

/** GET /api/validation/decision-report/{plan_id} */
export interface DecisionReportResponse {
  plan_id: string;
  validation_results: DecisionMetric[];
  summary: {
    metrics_compared: number;
    metrics_unavailable: number;
    metrics_errored: number;
    total_metrics: number;
    columns_affected: number;
    original_shape: { rows: number; columns: number };
    transformed_shape: { rows: number; columns: number };
  };
}
