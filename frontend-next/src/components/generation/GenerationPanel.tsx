"use client";

import { useEffect, useState } from "react";
import { useGenerate } from "@/hooks/api/useGenerate";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { useNavigationGuard } from "@/hooks/useNavigationGuard";
import { AlgorithmSelector } from "./AlgorithmSelector";
import { ParameterForm } from "./ParameterForm";
import { PresetManager } from "./PresetManager";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Wand2, Loader2, CheckCircle2, AlertTriangle, Circle } from "lucide-react";
import { toast } from "sonner";

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

  const isSequential = selectedAlgorithm === "PARS";

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
        onError: () => {
          toast.error("Generation failed", {
            description: "Check your connection or try fewer rows.",
          });
        },
      }
    );
  }

  return (
    <>
      <Tabs defaultValue="standard">
        <TabsList className="mb-4">
          <TabsTrigger value="standard">Standard (SDV)</TabsTrigger>
          <TabsTrigger value="advanced">Advanced (SynthCity)</TabsTrigger>
          <TabsTrigger value="llm" disabled>
            LLM-Guided
            <span className="ml-1.5 text-xs bg-muted px-1 rounded">Soon</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="standard" className="space-y-4">
          <AlgorithmSelector mode="sdv" />
          <PresetManager />
          <ParameterForm showSequenceParams={isSequential} />
        </TabsContent>

        <TabsContent value="advanced" className="space-y-4">
          <AlgorithmSelector mode="synthcity" />
          <PresetManager />
          <ParameterForm />
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
        <div className="flex items-center gap-2 text-sm text-green-600 mt-2">
          <CheckCircle2 className="h-4 w-4" />
          Synthetic data downloaded. You can generate again with different settings.
        </div>
      )}
      {generate.isError && (
        <div className="flex items-center gap-2 text-sm text-destructive mt-2">
          <AlertTriangle className="h-4 w-4" />
          Generation failed. Check your connection or try fewer rows.
        </div>
      )}

      <Button
        className="mt-4 w-full gap-2"
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
          <DialogHeader>
            <DialogTitle>Generating Synthetic Data</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <ol className="space-y-2" aria-label="Generation progress steps">
              {PROGRESS_STEPS.map((label, i) => {
                const done = activeStep > i;
                const active = activeStep === i;
                return (
                  <li key={label} className="flex items-center gap-3">
                    {done ? (
                      <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0" />
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
