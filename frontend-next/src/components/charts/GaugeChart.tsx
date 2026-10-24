"use client";

import { RadialBarChart, RadialBar, PolarAngleAxis, ResponsiveContainer, Tooltip } from "recharts";

interface GaugeChartProps {
  value: number | null; // 0–100; null = not available
  label: string;
  size?: number;
}

function scoreColor(v: number | null): string {
  if (v === null) return "#9ca3af";
  if (v >= 80) return "#10b981";
  if (v >= 60) return "#f59e0b";
  return "#f43f5e";
}

export function GaugeChart({ value, label, size = 120 }: GaugeChartProps) {
  const color = scoreColor(value);
  const data = [{ name: label, value: value ?? 0, fill: color }];

  return (
    <div className="flex flex-col items-center" style={{ width: size }}>
      <div className="relative" style={{ width: size, height: size }}>
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            cx="50%"
            cy="50%"
            innerRadius="72%"
            outerRadius="92%"
            data={data}
            startAngle={90}
            endAngle={-270}
          >
            {/* Fixed 0–100 scale; without it the single bar always fills the ring */}
            <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
            <RadialBar dataKey="value" background={{ fill: "hsl(var(--muted))" }} cornerRadius={10} />
            <Tooltip formatter={(v) => `${Number(v).toFixed(0)}/100`} />
          </RadialBarChart>
        </ResponsiveContainer>
        {/* Score in the middle of the ring */}
        <p className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center text-3xl font-bold leading-none tabular-nums" style={{ color }}>
          {value === null ? "n/a" : value.toFixed(0)}
          {value !== null && <span className="mt-1 text-[10px] font-medium text-muted-foreground">/ 100</span>}
        </p>
      </div>
      <p className="mt-2 text-sm font-semibold">{label}</p>
    </div>
  );
}

// Simple legend for quality categories
export function ScoreLegend() {
  return (
    <div className="flex gap-4 text-xs">
      <span className="flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1"><span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" /> ≥ 80 Good</span>
      <span className="flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1"><span className="w-2 h-2 rounded-full bg-amber-500 inline-block" /> 60–79 Fair</span>
      <span className="flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1"><span className="w-2 h-2 rounded-full bg-rose-500 inline-block" /> &lt; 60 Poor</span>
    </div>
  );
}
