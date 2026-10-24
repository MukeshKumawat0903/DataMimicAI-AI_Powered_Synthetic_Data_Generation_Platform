import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type {
  DecisionReportResponse,
  DiagnosticsBuildResponse,
  DiagnosticsInterpretation,
  PlanDecision,
  PlanExecution,
  PlanReview,
  TransformationPlan,
} from "@/lib/api/types";

// Step 1: build diagnostics from the dataset, then let the interpreter agent read them
export function useRunDiagnostics() {
  return useMutation({
    mutationFn: async (fileId: string) => {
      const built = await apiClient
        .post<DiagnosticsBuildResponse>(`/api/diagnostics/build`, { file_id: fileId, target_columns: null })
        .then((r) => r.data);
      const interpretation = await apiClient
        .post<{ interpretation: DiagnosticsInterpretation }>(`/api/diagnostics/interpret`, {
          diagnostics_input: built.diagnostics,
          rag_context: null,
        })
        .then((r) => r.data.interpretation);
      return { built, interpretation };
    },
  });
}

// Step 2: the planner agent proposes transformation plans (proposals only; nothing runs yet)
export function useCreatePlans() {
  return useMutation({
    mutationFn: (input: { built: DiagnosticsBuildResponse; interpretation: DiagnosticsInterpretation }) =>
      apiClient
        .post<{ proposed_plans: TransformationPlan[] }>(`/api/planner/create-plan`, {
          diagnostics: input.built.diagnostics,
          interpretation: input.interpretation,
          rag_context: null,
          signals: input.built.signals,
        })
        .then((r) => r.data.proposed_plans ?? []),
  });
}

// Step 3: a human approves or rejects each plan
export function useReviewPlan() {
  return useMutation({
    mutationFn: (input: { plan: TransformationPlan; decision: PlanDecision; notes: string }) =>
      apiClient
        .post<PlanReview>(`/api/approval/review-plan`, {
          plan: input.plan,
          decision: input.decision === "APPROVED" ? "approve" : "reject",
          reviewer_notes: input.notes || null,
        })
        .then((r) => r.data),
  });
}

// Step 4: the deterministic engine runs an approved plan and records before/after metrics
export function useExecutePlan() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (input: { planId: string; fileId: string }) =>
      apiClient
        .post<PlanExecution>(`/api/execution/execute-plan`, { plan_id: input.planId, file_id: input.fileId })
        .then((r) => r.data),
    onSuccess: (_, input) => {
      qc.invalidateQueries({ queryKey: ["validation-plans"] });
      qc.invalidateQueries({ queryKey: ["decision-report", input.planId] });
    },
  });
}

export function useValidationPlans() {
  return useQuery<string[]>({
    queryKey: ["validation-plans"],
    queryFn: () =>
      apiClient.get<{ plan_ids: string[] }>(`/api/validation/plans`).then((r) => r.data.plan_ids ?? []),
  });
}

export function useDecisionReport(planId: string | null) {
  return useQuery<DecisionReportResponse>({
    queryKey: ["decision-report", planId],
    queryFn: () =>
      apiClient.get<DecisionReportResponse>(`/api/validation/decision-report/${planId}`).then((r) => r.data),
    enabled: !!planId,
  });
}
