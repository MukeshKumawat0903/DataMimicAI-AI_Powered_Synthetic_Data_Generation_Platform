"use client";

import { SDV_ALGORITHM_INFO, SYNTHCITY_ALGORITHM_INFO } from "@/lib/constants";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { IconTile } from "@/components/ui/icon-tile";
import { Sparkles } from "lucide-react";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { Algorithm } from "@/lib/api/types";

interface AlgorithmSelectorProps {
  mode: "sdv" | "synthcity";
}

export function AlgorithmSelector({ mode }: AlgorithmSelectorProps) {
  const selectedAlgorithm = useWorkspaceStore((s) => s.selectedAlgorithm);
  const setSelectedAlgorithm = useWorkspaceStore((s) => s.setSelectedAlgorithm);

  const infoMap =
    mode === "sdv" ? SDV_ALGORITHM_INFO : SYNTHCITY_ALGORITHM_INFO;

  const info = infoMap[selectedAlgorithm] ?? null;

  return (
    <div className="space-y-2">
      <Label htmlFor="algorithm-select">Synthesis Algorithm</Label>
      <Select
        value={selectedAlgorithm}
        onValueChange={(v) => setSelectedAlgorithm(v as Algorithm)}
      >
        <SelectTrigger id="algorithm-select" className="w-full">
          <SelectValue placeholder="Select algorithm" />
        </SelectTrigger>
        <SelectContent>
          {Object.entries(infoMap).map(([key, val]) => (
            <SelectItem key={key} value={key}>
              <span className="font-medium">{key}</span>
              <span className="ml-2 text-xs text-muted-foreground">
                {val.use.replace("Best for: ", "").slice(0, 50)}…
              </span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {info && (
        <div className="flex items-start gap-3 rounded-xl bg-gradient-to-br from-violet-50 to-sky-50 px-4 py-3 ring-1 ring-inset ring-violet-200/70 dark:from-violet-500/10 dark:to-sky-500/10 dark:ring-violet-400/20">
          <IconTile icon={Sparkles} tone="violet" size="sm" className="mt-0.5" />
          <div className="space-y-0.5">
            <p className="text-sm font-medium">{info.desc}</p>
            <p className="text-xs text-muted-foreground">{info.use}</p>
          </div>
        </div>
      )}
    </div>
  );
}
