"use client";

import { RadialBarChart, RadialBar, ResponsiveContainer, Tooltip } from "recharts";

interface GaugeChartProps {
  value: number; // 0–100
  label: string;
  size?: number;
}

function scoreColor(v: number): string {
  if (v >= 80) return "#22c55e";
  if (v >= 60) return "#f59e0b";
  return "#ef4444";
}

export function GaugeChart({ value, label, size = 120 }: GaugeChartProps) {
  const color = scoreColor(value);
  const data = [{ name: label, value, fill: color }];

  return (
    <div className="flex flex-col items-center" style={{ width: size }}>
      <div style={{ width: size, height: size }}>
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            cx="50%"
            cy="50%"
            innerRadius="60%"
            outerRadius="80%"
            data={data}
            startAngle={90}
            endAngle={-270}
          >
            <RadialBar dataKey="value" background cornerRadius={4} />
            <Tooltip formatter={(v) => `${Number(v).toFixed(0)}/100`} />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-2xl font-bold leading-none mt-1" style={{ color }}>
        {value.toFixed(0)}
      </p>
      <p className="text-xs text-muted-foreground">{label}</p>
    </div>
  );
}

// Simple legend for quality categories
export function ScoreLegend() {
  return (
    <div className="flex gap-4 text-xs">
      <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-500 inline-block" /> ≥ 80 Good</span>
      <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-500 inline-block" /> 60–79 Fair</span>
      <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500 inline-block" /> &lt; 60 Poor</span>
    </div>
  );
}
