"use client";

import { SDV_ALGORITHM_INFO, SYNTHCITY_ALGORITHM_INFO } from "@/lib/constants";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { Card, CardContent } from "@/components/ui/card";
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
        <Card className="bg-muted/30">
          <CardContent className="pt-3 pb-3 space-y-0.5">
            <p className="text-sm">{info.desc}</p>
            <p className="text-xs text-muted-foreground">{info.use}</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
