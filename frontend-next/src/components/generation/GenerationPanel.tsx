"use client";

import { useEffect, useState } from "react";
import { useGenerate } from "@/hooks/api/useGenerate";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { useNavigationGuard } from "@/hooks/useNavigationGuard";
import { AlgorithmSelector } from "./AlgorithmSelector";
import { ParameterForm } from "./ParameterForm";
import { PresetManager } from "./PresetManager";
import { AlgorithmComparison } from "./AlgorithmComparison";
import { AutoMLPanel } from "./AutoMLPanel";
import { DownloadFormats } from "./DownloadFormats";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Wand2, Loader2, CheckCircle2, AlertTriangle, Circle, Boxes, Cpu, Bot } from "lucide-react";
import { Card } from "@/components/ui/card";
import { toast } from "sonner";
import { SDV_ALGORITHMS, SYNTHCITY_ALGORITHMS } from "@/lib/constants";
import type { Algorithm, AppError } from "@/lib/api/types";

function generationErrorMessage(err: unknown): string {
  const appErr = err as Partial<AppError> | null;
  if (!appErr?.status) return "Cannot reach the backend. Check that it is running.";
  return appErr.message || "Unexpected error. Please try again.";
}

const PROGRESS_STEPS = [
  "Preparing dataset…",
  "Fitting model…",
  "Generating samples…",
  "Saving & validating…",
] as const;

// cumulative durations (ms) before each step becomes active
const STEP_DELAYS_MS = [0, 12_000, 40_000, 70_000];

export function GenerationPanel() {
  const { fileId, pendingFeedback, generatorConfig, selectedAlgorithm } =
    useWorkspaceStore();
  const generatedFileId = useWorkspaceStore((s) => s.generatedFileId);
  const generate = useGenerate();

  const setSelectedAlgorithm = useWorkspaceStore((s) => s.setSelectedAlgorithm);
  const isSequential = selectedAlgorithm === "PARS";
  const isSynthCity = SYNTHCITY_ALGORITHMS.includes(selectedAlgorithm);

  // P1.3: Block accidental navigation while generating
  useNavigationGuard(generate.isPending);

  // P0.3: Simulated progress stepper
  const [activeStep, setActiveStep] = useState(0);
  const [progressPct, setProgressPct] = useState(0);

  useEffect(() => {
    if (!generate.isPending) {
      if (generate.isSuccess) {
        setActiveStep(PROGRESS_STEPS.length - 1);
        setProgressPct(100);
        toast.success("Synthetic data generated!", {
          description: "Head to the Validate tab to review and download.",
        });
      }
      const reset = setTimeout(() => {
        setActiveStep(0);
        setProgressPct(0);
      }, 1_500);
      return () => clearTimeout(reset);
    }

    // Reset when generation kicks off
    setActiveStep(0);
    setProgressPct(0);

    // Step timers
    const timers = STEP_DELAYS_MS.map((delay, i) =>
      setTimeout(() => setActiveStep(i), delay)
    );

    // Smooth progress up to 92%
    const tick = setInterval(
      () => setProgressPct((p) => (p < 92 ? p + 0.4 : p)),
      300
    );

    return () => {
      timers.forEach(clearTimeout);
      clearInterval(tick);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [generate.isPending]);

  function handleGenerate() {
    if (!fileId) return;
    generate.mutate(
      {
        file_id: fileId,
        feedback: pendingFeedback,
        generator_config: {
          ...generatorConfig,
          algorithm: selectedAlgorithm,
        },
      },
      {
        onError: (err) => {
          toast.error("Generation failed", { description: generationErrorMessage(err) });
        },
      }
    );
  }

  return (
    <>
      <Tabs
        defaultValue={isSynthCity ? "advanced" : "standard"}
        onValueChange={(tab) => {
          // Each tab has its own algorithm family; don't carry an SDV choice into SynthCity or back
          if (tab === "advanced" && !isSynthCity) setSelectedAlgorithm(SYNTHCITY_ALGORITHMS[0] as Algorithm);
          if (tab === "standard" && isSynthCity) setSelectedAlgorithm(SDV_ALGORITHMS[0] as Algorithm);
        }}
      >
        <TabsList className="mb-2 h-11 p-1">
          <TabsTrigger value="standard" className="gap-2 px-4">
            <Boxes className="h-4 w-4 text-sky-500" />
            Standard (SDV)
          </TabsTrigger>
          <TabsTrigger value="advanced" className="gap-2 px-4">
            <Cpu className="h-4 w-4 text-fuchsia-500" />
            Advanced (SynthCity)
          </TabsTrigger>
          <TabsTrigger value="llm" disabled className="gap-2 px-4">
            <Bot className="h-4 w-4" />
            LLM-Guided
            <span className="rounded-full bg-primary/10 px-1.5 text-[10px] font-semibold text-primary">Soon</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="standard">
          <Card className="gap-5 px-5 py-5">
            <AlgorithmSelector mode="sdv" />
            <AlgorithmComparison mode="sdv" />
            <PresetManager />
            <ParameterForm showSequenceParams={isSequential} />
          </Card>
        </TabsContent>

        <TabsContent value="advanced">
          <Card className="gap-5 px-5 py-5">
            <AlgorithmSelector mode="synthcity" />
            <AlgorithmComparison mode="synthcity" />
            <PresetManager />
            <ParameterForm />
            <AutoMLPanel />
          </Card>
        </TabsContent>

        <TabsContent value="llm">
          <p className="text-sm text-muted-foreground">
            LLM-guided generation is coming soon.
          </p>
        </TabsContent>
      </Tabs>

      {/* Generate button */}
      {/* P3.2: aria-live region for generation status */}
      <div aria-live="polite" aria-atomic="true" className="sr-only">
        {generate.isPending ? `Generating: ${PROGRESS_STEPS[activeStep]}` : ""}
        {generate.isSuccess ? "Generation complete." : ""}
        {generate.isError ? "Generation failed." : ""}
      </div>

      {generatedFileId && (
        <div className="mt-4 flex flex-wrap items-center gap-2 rounded-xl border border-emerald-500/30 bg-emerald-50 px-4 py-3 text-sm font-medium text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300">
          <CheckCircle2 className="h-4 w-4" />
          Synthetic data downloaded. You can generate again with different settings.
          {fileId && <DownloadFormats fileId={fileId} />}
        </div>
      )}
      {generate.isError && (
        <div className="mt-4 flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">
          <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
          <span>Generation failed: {generationErrorMessage(generate.error)}</span>
        </div>
      )}

      <Button
        size="lg"
        className="mt-4 w-full gap-2 text-base"
        onClick={handleGenerate}
        disabled={generate.isPending || !fileId}
      >
        {generate.isPending ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Wand2 className="h-4 w-4" />
        )}
        {generate.isPending ? "Generating…" : "Generate Synthetic Data"}
      </Button>

      {/* P0.3: Progress overlay with simulated stepper */}
      <Dialog open={generate.isPending} onOpenChange={() => {}}>
        <DialogContent className="sm:max-w-md" showCloseButton={false}>
          <DialogHeader className="flex-row items-center gap-3">
            <span className="bg-brand-gradient flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-white shadow-md shadow-indigo-500/30">
              <Wand2 className="h-5 w-5 animate-pulse" />
            </span>
            <div>
              <DialogTitle>Generating Synthetic Data</DialogTitle>
              <p className="text-xs text-muted-foreground">Training {selectedAlgorithm} on your data</p>
            </div>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <ol className="space-y-2" aria-label="Generation progress steps">
              {PROGRESS_STEPS.map((label, i) => {
                const done = activeStep > i;
                const active = activeStep === i;
                return (
                  <li key={label} className="flex items-center gap-3">
                    {done ? (
                      <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0" />
                    ) : active ? (
                      <Loader2 className="h-4 w-4 animate-spin text-primary shrink-0" />
                    ) : (
                      <Circle className="h-4 w-4 text-muted-foreground/40 shrink-0" />
                    )}
                    <span
                      className={
                        done
                          ? "text-sm text-muted-foreground line-through"
                          : active
                          ? "text-sm font-medium"
                          : "text-sm text-muted-foreground/60"
                      }
                    >
                      {label}
                    </span>
                  </li>
                );
              })}
            </ol>
            <Progress value={progressPct} className="h-2" />
            <p className="text-xs text-muted-foreground">
              This may take 30 seconds to several minutes depending on dataset size.
              If the backend is warming up (cold start), please wait up to 60 seconds.
            </p>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
