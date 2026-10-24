"use client";

import { useState } from "react";
import { useCreatePlans, useExecutePlan, useReviewPlan, useRunDiagnostics } from "@/hooks/api/useAgents";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { AlertTriangle, CheckCircle2, Circle, Loader2, Play, ThumbsDown, ThumbsUp, ListChecks } from "lucide-react";
import { IconTile } from "@/components/ui/icon-tile";
import { toast } from "sonner";
import { cn, humanize } from "@/lib/utils";
import type { AppError, PlanDecision, TransformationPlan } from "@/lib/api/types";

const SEVERITY_CLASS: Record<string, string> = {
  high: "bg-red-100 text-red-700 border-red-200",
  medium: "bg-amber-100 text-amber-700 border-amber-200",
  low: "bg-blue-100 text-blue-700 border-blue-200",
};

const errorText = (e: unknown) => (e as Partial<AppError>)?.message ?? "Please try again.";

function StepHeader({ n, title, done, active }: { n: number; title: string; done: boolean; active: boolean }) {
  return (
    <div className={cn("flex items-center gap-2 text-sm font-medium", !active && !done && "text-muted-foreground")}>
      {done ? <CheckCircle2 className="h-4 w-4 text-green-600" /> : <Circle className="h-4 w-4" />}
      Step {n}: {title}
    </div>
  );
}

export function ActionPlanner({ onOpenReport }: { onOpenReport: (planId: string) => void }) {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const planner = useWorkspaceStore((s) => s.planner);
  const updatePlanner = useWorkspaceStore((s) => s.updatePlanner);

  const diagnose = useRunDiagnostics();
  const createPlans = useCreatePlans();
  const review = useReviewPlan();
  const execute = useExecutePlan();
  const [notes, setNotes] = useState<Record<string, string>>({});

  const { built, interpretation, plans } = planner;
  const approvedCount = Object.values(planner.reviews).filter((r) => r.status === "APPROVED").length;

  function runDiagnostics() {
    if (!fileId) return;
    diagnose.mutate(fileId, {
      // New diagnostics invalidate any plans made from the old ones
      onSuccess: (res) => updatePlanner({ ...res, plans: undefined, reviews: {}, executions: {} }),
      onError: (e) => toast.error("Diagnostics failed", { description: errorText(e) }),
    });
  }

  function generatePlans() {
    if (!built || !interpretation) return;
    createPlans.mutate(
      { built, interpretation },
      {
        onSuccess: (proposed) => updatePlanner({ plans: proposed, reviews: {}, executions: {} }),
        onError: (e) => toast.error("Planning failed", { description: errorText(e) }),
      }
    );
  }

  function decide(plan: TransformationPlan, decision: PlanDecision) {
    review.mutate(
      { plan, decision, notes: notes[plan.plan_id] ?? "" },
      {
        onSuccess: (rec) => updatePlanner((p) => ({ reviews: { ...p.reviews, [plan.plan_id]: rec } })),
        onError: (e) => toast.error("Review failed", { description: errorText(e) }),
      }
    );
  }

  function runPlan(planId: string) {
    if (!fileId) return;
    execute.mutate(
      { planId, fileId },
      {
        onSuccess: (res) => {
          updatePlanner((p) => ({ executions: { ...p.executions, [planId]: res } }));
          if (res.execution_status === "SUCCESS") toast.success(`Plan ${planId} executed`);
          else toast.error(`Plan ${planId} failed`, { description: res.error ?? undefined });
        },
        onError: (e) => toast.error("Execution failed", { description: errorText(e) }),
      }
    );
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-start gap-3 py-3">
        <IconTile icon={ListChecks} tone="violet" />
        <div className="space-y-1">
          <CardTitle className="text-base">Action Planner</CardTitle>
          <p className="text-sm text-muted-foreground">
            AI agents diagnose the data and propose transformation plans. Nothing changes until you approve a
            plan; execution then runs a deterministic engine and records before/after metrics.
          </p>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Step 1: Diagnose */}
        <section className="space-y-3">
          <div className="flex items-center justify-between gap-2">
            <StepHeader n={1} title="Diagnose" done={!!interpretation} active />
            <Button size="sm" variant={interpretation ? "outline" : "default"} onClick={runDiagnostics} disabled={diagnose.isPending}>
              {diagnose.isPending && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1" />}
              {interpretation ? "Re-run diagnostics" : "Run diagnostics"}
            </Button>
          </div>

          {built && interpretation && (
            <div className="space-y-3 pl-6">
              <div className="flex flex-wrap items-center gap-2 text-sm">
                <span className="font-medium">{built.diagnostics.summary.total_issues} issues</span>
                <span className="text-muted-foreground">
                  ({built.diagnostics.summary.high_severity_count} high · {built.diagnostics.summary.medium_severity_count} medium ·{" "}
                  {built.diagnostics.summary.low_severity_count} low)
                </span>
                <span className="mx-1 text-muted-foreground">|</span>
                <span>
                  Assessment: <span className="font-medium">{humanize(interpretation.overall_assessment)}</span>
                </span>
                <Badge variant="outline" className="text-xs">
                  {interpretation.confidence} confidence
                </Badge>
              </div>

              {interpretation.dominant_issue_patterns.length > 0 && (
                <div className="flex flex-wrap items-center gap-1.5 text-sm">
                  <span className="text-muted-foreground">Main patterns:</span>
                  {interpretation.dominant_issue_patterns.map((p) => (
                    <Badge key={p} variant="secondary">
                      {humanize(p)}
                    </Badge>
                  ))}
                </div>
              )}

              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Issue</TableHead>
                    <TableHead>Columns</TableHead>
                    <TableHead>Metric</TableHead>
                    <TableHead>Severity</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {built.diagnostics.diagnostics.map((d, i) => (
                    <TableRow key={i}>
                      <TableCell>{humanize(d.issue_type)}</TableCell>
                      <TableCell className="font-mono text-xs">{d.affected_columns.join(", ")}</TableCell>
                      <TableCell className="text-xs">
                        {humanize(d.metric_name)} = {Number(d.metric_value).toFixed(2)}
                      </TableCell>
                      <TableCell>
                        <span className={cn("rounded border px-1.5 py-0.5 text-xs", SEVERITY_CLASS[d.severity])}>
                          {d.severity}
                        </span>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </section>

        {/* Step 2: Plan */}
        <section className="space-y-3">
          <div className="flex items-center justify-between gap-2">
            <StepHeader n={2} title="Propose plans" done={!!plans} active={!!interpretation} />
            <Button
              size="sm"
              variant={plans ? "outline" : "default"}
              onClick={generatePlans}
              disabled={!interpretation || createPlans.isPending}
            >
              {createPlans.isPending && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1" />}
              {plans ? "Regenerate plans" : "Generate plans"}
            </Button>
          </div>
          {plans && plans.length === 0 && (
            <p className="pl-6 text-sm text-muted-foreground">The planner found nothing worth transforming.</p>
          )}
        </section>

        {/* Steps 3 + 4: Review and execute, per plan */}
        {plans && plans.length > 0 && (
          <section className="space-y-3">
            <StepHeader n={3} title="Review" done={Object.keys(planner.reviews).length === plans.length} active />
            <StepHeader n={4} title="Execute" done={Object.keys(planner.executions).length > 0} active={approvedCount > 0} />

            {plans.map((plan) => {
              const rec = planner.reviews[plan.plan_id];
              const run = planner.executions[plan.plan_id];
              return (
                <div key={plan.plan_id} className="ml-6 rounded-lg border p-3 space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm font-medium">{plan.plan_id}</span>
                      {plan.applicable_issue_patterns.map((p) => (
                        <Badge key={p} variant="secondary" className="text-xs">
                          {humanize(p)}
                        </Badge>
                      ))}
                    </div>
                    {rec ? (
                      <Badge variant={rec.status === "APPROVED" ? "default" : "destructive"}>{rec.status}</Badge>
                    ) : (
                      <Badge variant="outline">Pending review</Badge>
                    )}
                  </div>

                  <ol className="space-y-1.5 text-sm list-decimal pl-5">
                    {plan.proposed_transformations.map((t, i) => (
                      <li key={i}>
                        <span className="font-medium">{humanize(t.transformation)}</span> on{" "}
                        <span className="font-mono text-xs">{t.target_columns.join(", ")}</span>
                        <p className="text-xs text-muted-foreground">{t.rationale}</p>
                      </li>
                    ))}
                  </ol>

                  {(plan.assumptions.length > 0 || plan.risks.length > 0) && (
                    <div className="grid gap-2 text-xs sm:grid-cols-2">
                      {plan.assumptions.length > 0 && (
                        <div>
                          <p className="font-medium">Assumptions</p>
                          <ul className="list-disc pl-4 text-muted-foreground">
                            {plan.assumptions.map((a) => <li key={a}>{a}</li>)}
                          </ul>
                        </div>
                      )}
                      {plan.risks.length > 0 && (
                        <div>
                          <p className="font-medium flex items-center gap-1">
                            <AlertTriangle className="h-3 w-3 text-amber-500" /> Risks
                          </p>
                          <ul className="list-disc pl-4 text-muted-foreground">
                            {plan.risks.map((r) => <li key={r}>{r}</li>)}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}

                  {!rec && (
                    <div className="space-y-2">
                      <textarea
                        aria-label={`Reviewer notes for ${plan.plan_id}`}
                        placeholder="Reviewer notes (optional)"
                        className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                        rows={2}
                        value={notes[plan.plan_id] ?? ""}
                        onChange={(e) => setNotes((n) => ({ ...n, [plan.plan_id]: e.target.value }))}
                      />
                      <div className="flex gap-2">
                        <Button size="sm" onClick={() => decide(plan, "APPROVED")} disabled={review.isPending} className="gap-1">
                          <ThumbsUp className="h-3.5 w-3.5" /> Approve
                        </Button>
                        <Button size="sm" variant="outline" onClick={() => decide(plan, "REJECTED")} disabled={review.isPending} className="gap-1">
                          <ThumbsDown className="h-3.5 w-3.5" /> Reject
                        </Button>
                      </div>
                    </div>
                  )}

                  {rec?.status === "APPROVED" && (
                    <div className="flex flex-wrap items-center gap-2">
                      <Button size="sm" onClick={() => runPlan(plan.plan_id)} disabled={execute.isPending} className="gap-1">
                        {execute.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
                        {run ? "Execute again" : "Execute plan"}
                      </Button>
                      {run && (
                        <>
                          <span className={cn("text-sm", run.execution_status === "SUCCESS" ? "text-green-600" : "text-destructive")}>
                            {run.execution_status === "SUCCESS"
                              ? `Applied: ${run.applied_transformations.map(humanize).join(", ")}`
                              : `Failed: ${run.error ?? "unknown error"}`}
                          </span>
                          {run.validation_available && (
                            <Button size="sm" variant="outline" onClick={() => onOpenReport(run.validation_plan_id ?? plan.plan_id)}>
                              View decision report
                            </Button>
                          )}
                        </>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </section>
        )}
      </CardContent>
    </Card>
  );
}
