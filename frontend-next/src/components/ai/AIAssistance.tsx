"use client";

import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Bot, FileBarChart, ListChecks } from "lucide-react";
import { LLMExplanationPanel } from "@/components/eda/LLMExplanationPanel";
import { ActionPlanner } from "./ActionPlanner";
import { DecisionReport } from "./DecisionReport";

/** AI proposes, a human approves, a deterministic engine executes. */
export function AIAssistance() {
  const [tab, setTab] = useState("diagnostics");
  const [reportPlanId, setReportPlanId] = useState<string | null>(null);

  return (
    <Tabs value={tab} onValueChange={(v) => setTab(v as string)}>
      <TabsList className="h-auto bg-card p-1.5 shadow-sm">
        <TabsTrigger value="diagnostics" className="h-8 gap-1.5 text-xs">
          <Bot className="h-3.5 w-3.5 text-violet-500" />
          Diagnostics
        </TabsTrigger>
        <TabsTrigger value="planner" className="h-8 gap-1.5 text-xs">
          <ListChecks className="h-3.5 w-3.5 text-indigo-500" />
          Action Planner
        </TabsTrigger>
        <TabsTrigger value="report" className="h-8 gap-1.5 text-xs">
          <FileBarChart className="h-3.5 w-3.5 text-sky-500" />
          Decision Report
        </TabsTrigger>
      </TabsList>
      <TabsContent value="diagnostics" className="mt-3">
        <LLMExplanationPanel />
      </TabsContent>
      <TabsContent value="planner" className="mt-3">
        <ActionPlanner
          onOpenReport={(planId) => {
            setReportPlanId(planId);
            setTab("report");
          }}
        />
      </TabsContent>
      <TabsContent value="report" className="mt-3">
        <DecisionReport planId={reportPlanId} onPlanChange={setReportPlanId} />
      </TabsContent>
    </Tabs>
  );
}
