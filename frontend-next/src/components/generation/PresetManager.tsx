"use client";

import { useState, useEffect } from "react";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Save, FolderOpen, Trash2 } from "lucide-react";
import type { Algorithm } from "@/lib/api/types";

interface Preset {
  name: string;
  algorithm: Algorithm;
  num_rows: number;
  epochs?: number;
}

const STORAGE_KEY = "datamimicai_presets";

function loadPresets(): Preset[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Preset[]) : [];
  } catch {
    return [];
  }
}

function savePresets(presets: Preset[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(presets));
}

export function PresetManager() {
  const { selectedAlgorithm, generatorConfig, setGeneratorConfig, setSelectedAlgorithm } =
    useWorkspaceStore();

  const [presets, setPresets] = useState<Preset[]>([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [presetName, setPresetName] = useState("");

  useEffect(() => {
    setPresets(loadPresets());
  }, []);

  function handleSave() {
    if (!presetName.trim()) return;
    const newPreset: Preset = {
      name: presetName.trim(),
      algorithm: selectedAlgorithm,
      num_rows: generatorConfig.num_rows,
      epochs: generatorConfig.epochs,
    };
    const updated = [...presets.filter((p) => p.name !== newPreset.name), newPreset];
    setPresets(updated);
    savePresets(updated);
    setPresetName("");
    setDialogOpen(false);
  }

  function handleLoad(preset: Preset) {
    setSelectedAlgorithm(preset.algorithm);
    setGeneratorConfig({ num_rows: preset.num_rows, epochs: preset.epochs });
  }

  function handleDelete(name: string) {
    const updated = presets.filter((p) => p.name !== name);
    setPresets(updated);
    savePresets(updated);
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs text-muted-foreground">Saved presets:</span>
        {presets.map((p) => (
          <Badge
            key={p.name}
            variant="outline"
            className="cursor-pointer hover:bg-accent flex items-center gap-1"
            onClick={() => handleLoad(p)}
          >
            <FolderOpen className="h-3 w-3" />
            {p.name}
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleDelete(p.name);
              }}
              className="ml-1 text-muted-foreground hover:text-destructive"
              aria-label={`Delete preset ${p.name}`}
            >
              <Trash2 className="h-2.5 w-2.5" />
            </button>
          </Badge>
        ))}
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-xs"
          onClick={() => setDialogOpen(true)}
        >
          <Save className="h-3 w-3 mr-1" />
          Save current
        </Button>
      </div>

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Save Preset</DialogTitle>
          </DialogHeader>
          <Input
            placeholder="Preset name"
            value={presetName}
            onChange={(e) => setPresetName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSave()}
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleSave} disabled={!presetName.trim()}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
