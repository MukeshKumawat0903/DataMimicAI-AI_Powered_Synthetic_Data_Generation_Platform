"use client";

import { useEffect } from "react";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { PRESET_CONFIGS } from "@/lib/constants";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronDown, AlertTriangle } from "lucide-react";

const schema = z.object({
  num_rows: z.number({ message: "Must be a number" }).int().min(1).max(100000),
  epochs: z.number({ message: "Must be a number" }).int().min(10).max(2000).optional(),
  num_sequences: z.number().int().min(1).optional(),
  sequence_length: z.number().int().min(1).optional(),
});

type FormValues = z.infer<typeof schema>;

interface ParameterFormProps {
  showSequenceParams?: boolean;
}

export function ParameterForm({ showSequenceParams = false }: ParameterFormProps) {
  const generatorConfig = useWorkspaceStore((s) => s.generatorConfig);
  const setGeneratorConfig = useWorkspaceStore((s) => s.setGeneratorConfig);

  const { register, control, handleSubmit, setValue, watch, formState: { errors } } =
    useForm<FormValues>({
      resolver: zodResolver(schema),
      defaultValues: {
        num_rows: generatorConfig.num_rows ?? 1000,
        epochs: generatorConfig.epochs ?? 300,
        num_sequences: generatorConfig.num_sequences,
        sequence_length: generatorConfig.sequence_length,
      },
    });

  const numRows = watch("num_rows");

  // Sync form → store on any change
  useEffect(() => {
    const sub = watch((vals) => {
      setGeneratorConfig({
        num_rows: vals.num_rows ?? 1000,
        epochs: vals.epochs,
        num_sequences: vals.num_sequences,
        sequence_length: vals.sequence_length,
      });
    });
    return () => sub.unsubscribe();
  }, [watch, setGeneratorConfig]);

  function applyPreset(name: string) {
    const p = PRESET_CONFIGS[name];
    if (!p) return;
    setValue("num_rows", p.num_rows);
    setValue("epochs", p.epochs);
  }

  return (
    <form onSubmit={handleSubmit(() => {})} className="space-y-4">
      {/* Preset buttons */}
      <div className="flex gap-2 flex-wrap">
        {Object.keys(PRESET_CONFIGS).map((name) => (
          <Button
            key={name}
            type="button"
            variant="outline"
            size="sm"
            onClick={() => applyPreset(name)}
          >
            {name}
          </Button>
        ))}
      </div>

      {/* num_rows */}
      <div className="space-y-1">
        <Label htmlFor="num_rows">Number of rows to generate</Label>
        <Controller
          name="num_rows"
          control={control}
          render={({ field }) => (
            <Input
              id="num_rows"
              type="number"
              min={1}
              max={100000}
              {...field}
              onChange={(e) => field.onChange(parseInt(e.target.value, 10) || 0)}
            />
          )}
        />
        {errors.num_rows && (
          <p className="text-xs text-destructive">{errors.num_rows.message}</p>
        )}
        {numRows > 50000 && (
          <div className="flex items-center gap-1 text-xs text-amber-600 mt-1">
            <AlertTriangle className="h-3.5 w-3.5" />
            Generating {numRows.toLocaleString()} rows may require significant memory and time.
          </div>
        )}
      </div>

      {/* Advanced options (epochs) */}
      <Collapsible>
        <CollapsibleTrigger className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
          <ChevronDown className="h-4 w-4" />
          Advanced options
        </CollapsibleTrigger>
        <CollapsibleContent className="pt-3 space-y-3">
          <div className="space-y-1">
            <Label htmlFor="epochs">Training epochs</Label>
            <Controller
              name="epochs"
              control={control}
              render={({ field }) => (
                <Input
                  id="epochs"
                  type="number"
                  min={10}
                  max={2000}
                  {...field}
                  value={field.value ?? ""}
                  onChange={(e) => field.onChange(parseInt(e.target.value, 10) || undefined)}
                />
              )}
            />
            {errors.epochs && (
              <p className="text-xs text-destructive">{errors.epochs.message}</p>
            )}
          </div>

          {showSequenceParams && (
            <>
              <div className="space-y-1">
                <Label htmlFor="num_sequences">Number of sequences</Label>
                <Input
                  id="num_sequences"
                  type="number"
                  min={1}
                  {...register("num_sequences", { valueAsNumber: true })}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="sequence_length">Sequence length</Label>
                <Input
                  id="sequence_length"
                  type="number"
                  min={1}
                  {...register("sequence_length", { valueAsNumber: true })}
                />
              </div>
            </>
          )}
        </CollapsibleContent>
      </Collapsible>
    </form>
  );
}
