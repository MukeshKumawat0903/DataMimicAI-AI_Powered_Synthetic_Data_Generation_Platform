import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { LLMExplanationOptions, LLMExplanationResponse, LLMScopes } from "@/lib/api/types";

// Backend response shape from POST /llm/explain
interface BackendExplanationResponse {
  explanation: string;
  validated: boolean;
  validation_report: {
    length_valid?: boolean;
    no_hallucination?: boolean;
    scope_aligned?: boolean;
    final_verdict?: "PASS" | "FAIL";
  };
  metadata?: Record<string, unknown>;
}

function mapResponse(res: BackendExplanationResponse): LLMExplanationResponse {
  const report = res.validation_report ?? {};
  const checks = [
    { label: "Length within limits", passed: report.length_valid ?? false },
    { label: "No unsupported facts", passed: report.no_hallucination ?? false },
    { label: "Matches the chosen scope", passed: report.scope_aligned ?? false },
  ];
  const passed = checks.filter((c) => c.passed).length;
  return {
    explanation: res.explanation,
    validated: res.validated,
    validation_pct: res.validated ? 100 : Math.round((passed / checks.length) * 100),
    checks,
    source: res.metadata?.explanation_source === "facts_summary" ? "facts_summary" : "llm",
  };
}

export function useLLMScopes() {
  return useQuery<LLMScopes>({
    queryKey: ["llm-scopes"],
    queryFn: () => apiClient.get<LLMScopes>(`/llm/scopes`).then((r) => r.data),
    staleTime: Infinity,
  });
}

export function useLLMExplanation(fileId: string | null, options: LLMExplanationOptions, enabled = false) {
  return useQuery<LLMExplanationResponse>({
    queryKey: ["llm-explanation", fileId, options],
    queryFn: () =>
      apiClient
        .post<BackendExplanationResponse>(`/llm/explain`, {
          file_id: fileId,
          scope: options.scope,
          tone: options.tone,
          ...(options.columns?.length ? { columns: options.columns } : {}),
        })
        .then((r) => mapResponse(r.data)),
    enabled: !!fileId && enabled,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  });
}
