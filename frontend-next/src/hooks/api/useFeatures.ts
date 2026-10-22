import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { FeatureSuggestion, FeaturesResponse } from "@/lib/api/types";

// Backend response shape from POST /eda/context-aware-suggestions
interface BackendSuggestion {
  column: string;
  transformation: string;
  reason: string;
  confidence?: number; // utility suggestions
  risk_level?: "high" | "medium" | "low"; // privacy suggestions
}

interface BackendFeaturesResponse {
  file_id?: string;
  utility_suggestions?: BackendSuggestion[];
  privacy_suggestions?: BackendSuggestion[];
  conflicts?: Array<{ column: string; recommendation?: string }>;
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

const toSuggestion = (s: BackendSuggestion): FeatureSuggestion => ({
  column: s.column,
  suggestion_type: s.transformation,
  description: s.reason,
  impact: toImpact(s),
  resolved: false,
});

function mapResponse(res: BackendFeaturesResponse): FeaturesResponse {
  return {
    file_id: res.file_id,
    utility_suggestions: (res.utility_suggestions ?? []).map(toSuggestion),
    privacy_suggestions: (res.privacy_suggestions ?? []).map(toSuggestion),
    conflicts: (res.conflicts ?? []).map((c) => ({
      column: c.column,
      reason: c.recommendation,
    })),
    conflict_summary: res.conflict_summary,
    non_conflicting_utility: (res.non_conflicting_utility ?? []).map(toSuggestion),
    non_conflicting_privacy: (res.non_conflicting_privacy ?? []).map(toSuggestion),
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
