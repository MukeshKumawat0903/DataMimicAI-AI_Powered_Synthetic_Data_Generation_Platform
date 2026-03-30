import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { LLMExplanationResponse } from "@/lib/api/types";

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
    report.length_valid ?? false,
    report.no_hallucination ?? false,
    report.scope_aligned ?? false,
  ];
  const passedCount = checks.filter(Boolean).length;
  // If fully validated use 100, otherwise score by what passed
  const validation_pct = res.validated ? 100 : Math.round((passedCount / 3) * 100);

  const STEP_LABELS = [
    "Loading dataset",
    "Detecting issues",
    "Gathering context",
    "Querying LLM",
    "Parsing response",
    "Validating facts",
    "Preparing output",
  ];

  return {
    explanation: res.explanation,
    validation_pct,
    steps: STEP_LABELS.map((label, i) => ({
      step: i + 1,
      label,
      status: "done" as const,
    })),
  };
}

export function useLLMExplanation(fileId: string | null, enabled = false) {
  return useQuery<LLMExplanationResponse>({
    queryKey: ["llm-explanation", fileId],
    queryFn: () =>
      apiClient
        .post<BackendExplanationResponse>(`/llm/explain`, {
          file_id: fileId,
          scope: "dataset_overview",
          tone: "clear",
        })
        .then((r) => mapResponse(r.data)),
    enabled: !!fileId && enabled,
    staleTime: 0,
    retry: 1,
  });
}
