"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle, FlaskConical } from "lucide-react";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { useLoadDemo } from "@/hooks/api/useUpload";
import type { AppError } from "@/lib/api/types";

const ALGORITHMS = [
  {
    value: "CTGAN",
    label: "CTGAN",
    description: "Best for large, complex, mixed-type tabular datasets.",
  },
  {
    value: "GaussianCopula",
    label: "GaussianCopula",
    description: "Fast and robust for small-to-medium numeric datasets.",
  },
  {
    value: "TVAE",
    label: "TVAE",
    description: "Neural network-based; handles complex relationships.",
  },
  {
    value: "PARS",
    label: "PARS",
    description: "Pattern-based for sequential / time-series data.",
  },
] as const;

export function DemoModeSelector() {
  const demoAlgorithm = useWorkspaceStore((s) => s.demoAlgorithm);
  const setDemoAlgorithm = useWorkspaceStore((s) => s.setDemoAlgorithm);

  const { mutate: loadDemo, isPending, error, isSuccess } = useLoadDemo();

  const selectedInfo = ALGORITHMS.find((a) => a.value === demoAlgorithm);
  const loadError = error as AppError | null;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <FlaskConical className="h-4 w-4 text-primary" />
          Demo Mode
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Algorithm selector */}
        <div className="space-y-1.5">
          <label
            htmlFor="demo-algo"
            className="text-sm font-medium text-muted-foreground"
          >
            Select Algorithm
          </label>
          <Select
            value={demoAlgorithm}
            onValueChange={(v) => { if (v) setDemoAlgorithm(v); }}
            disabled={isPending}
          >
            <SelectTrigger id="demo-algo" className="w-full">
              <SelectValue placeholder="Pick an algorithm" />
            </SelectTrigger>
            <SelectContent>
              {ALGORITHMS.map((algo) => (
                <SelectItem key={algo.value} value={algo.value}>
                  {algo.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {selectedInfo && (
            <p className="text-xs text-muted-foreground">
              {selectedInfo.description}
            </p>
          )}
        </div>

        {/* Feedback */}
        {loadError && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{loadError.message}</AlertDescription>
          </Alert>
        )}
        {isSuccess && (
          <Alert className="border-green-500/50 bg-green-50 dark:bg-green-950/20">
            <AlertDescription className="text-green-700 dark:text-green-400 text-sm">
              Demo data loaded!
            </AlertDescription>
          </Alert>
        )}

        <Button
          className="w-full"
          variant="outline"
          onClick={() => loadDemo(demoAlgorithm)}
          disabled={isPending}
        >
          {isPending ? `Loading ${demoAlgorithm} demo…` : "Load Demo Data"}
        </Button>
      </CardContent>
    </Card>
  );
}
