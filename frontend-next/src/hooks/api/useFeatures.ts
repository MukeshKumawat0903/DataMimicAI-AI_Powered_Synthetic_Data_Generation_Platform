import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { FeatureSuggestion, FeaturesResponse } from "@/lib/api/types";

// Backend response shape from POST /eda/context-aware-suggestions
interface BackendSuggestion {
  column: string;
  transformation: string;
  reason: string;
  category?: "utility" | "privacy";
  params?: Record<string, unknown>;
  metrics?: Record<string, unknown>;
  confidence?: number; // utility suggestions
  risk_level?: "high" | "medium" | "low"; // privacy suggestions
}

interface BackendFeaturesResponse {
  file_id?: string;
  utility_suggestions?: BackendSuggestion[];
  privacy_suggestions?: BackendSuggestion[];
  conflicts?: Array<{
    column: string;
    severity?: string;
    recommendation?: string;
    utility_suggestion?: BackendSuggestion;
    privacy_suggestion?: BackendSuggestion;
  }>;
  conflict_summary?: Record<string, unknown>;
  non_conflicting_utility?: BackendSuggestion[];
  non_conflicting_privacy?: BackendSuggestion[];
  metadata?: Record<string, unknown>;
}

function toImpact(s: BackendSuggestion): FeatureSuggestion["impact"] {
  if (s.risk_level) return s.risk_level;
  const c = s.confidence ?? 0;
  return c >= 0.75 ? "high" : c >= 0.5 ? "medium" : "low";
}

const toSuggestion = (s: BackendSuggestion, category: "utility" | "privacy"): FeatureSuggestion => ({
  column: s.column,
  suggestion_type: s.transformation,
  description: s.reason,
  impact: toImpact(s),
  resolved: false,
  category: s.category ?? category,
  params: s.params ?? {},
  metrics: s.metrics,
});

function mapResponse(res: BackendFeaturesResponse): FeaturesResponse {
  return {
    file_id: res.file_id,
    utility_suggestions: (res.utility_suggestions ?? []).map((s) => toSuggestion(s, "utility")),
    privacy_suggestions: (res.privacy_suggestions ?? []).map((s) => toSuggestion(s, "privacy")),
    conflicts: (res.conflicts ?? []).map((c) => ({
      column: c.column,
      severity: c.severity,
      reason: c.recommendation,
      utility: c.utility_suggestion ? toSuggestion(c.utility_suggestion, "utility") : undefined,
      privacy: c.privacy_suggestion ? toSuggestion(c.privacy_suggestion, "privacy") : undefined,
    })),
    conflict_summary: res.conflict_summary,
    non_conflicting_utility: (res.non_conflicting_utility ?? []).map((s) => toSuggestion(s, "utility")),
    non_conflicting_privacy: (res.non_conflicting_privacy ?? []).map((s) => toSuggestion(s, "privacy")),
    metadata: res.metadata,
  };
}

export function useFeatures(fileId: string | null) {
  return useQuery<FeaturesResponse>({
    queryKey: ["features", fileId],
    queryFn: () =>
      apiClient
        // context-aware-suggestions returns utility_suggestions, privacy_suggestions, and conflicts
        .post<BackendFeaturesResponse>(`/eda/context-aware-suggestions`, null, {
          params: { file_id: fileId },
        })
        .then((r) => mapResponse(r.data)),
    enabled: !!fileId,
    staleTime: 5 * 60 * 1000,
  });
}
