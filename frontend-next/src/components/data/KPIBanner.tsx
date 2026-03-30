"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TriangleAlert, TrendingDown, TrendingUp, Minus } from "lucide-react";
import { cn } from "@/lib/utils";
import type { KPISummary } from "@/lib/api/types";

interface KPIBannerProps {
  summary: KPISummary;
}

interface MetricCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  variant?: "default" | "positive" | "negative" | "warning";
  icon?: React.ReactNode;
}

function MetricCard({
  label,
  value,
  subtext,
  variant = "default",
  icon,
}: MetricCardProps) {
  return (
    <Card className="flex-1 min-w-[140px]">
      <CardContent className="pt-4 pb-4 px-4">
        <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide mb-1">
          {label}
        </p>
        <div className="flex items-end gap-2">
          <span
            className={cn("text-2xl font-bold tabular-nums", {
              "text-green-600 dark:text-green-400": variant === "positive",
              "text-red-600 dark:text-red-400": variant === "negative",
              "text-amber-600 dark:text-amber-400": variant === "warning",
            })}
          >
            {value}
          </span>
          {icon}
        </div>
        {subtext && (
          <p className="text-xs text-muted-foreground mt-0.5">{subtext}</p>
        )}
      </CardContent>
    </Card>
  );
}

export function KPIBanner({ summary }: KPIBannerProps) {
  const {
    pct_transformed,
    avg_skewness_change,
    avg_outlier_change,
    improved_count,
    worsened_count,
    risky_columns,
  } = summary;

  const skewIcon =
    avg_skewness_change < -0.1 ? (
      <TrendingDown className="h-4 w-4 text-green-500" />
    ) : avg_skewness_change > 0.1 ? (
      <TrendingUp className="h-4 w-4 text-red-500" />
    ) : (
      <Minus className="h-4 w-4 text-muted-foreground" />
    );

  const outlierIcon =
    avg_outlier_change < -1 ? (
      <TrendingDown className="h-4 w-4 text-green-500" />
    ) : avg_outlier_change > 1 ? (
      <TrendingUp className="h-4 w-4 text-red-500" />
    ) : (
      <Minus className="h-4 w-4 text-muted-foreground" />
    );

  return (
    <div className="space-y-3">
      {/* Metric row */}
      <div className="flex flex-wrap gap-3">
        <MetricCard
          label="Transformed"
          value={`${pct_transformed.toFixed(0)}%`}
          subtext="of columns"
        />
        <MetricCard
          label="Improved"
          value={improved_count}
          variant={improved_count > 0 ? "positive" : "default"}
          subtext="columns"
        />
        <MetricCard
          label="Worsened"
          value={worsened_count}
          variant={worsened_count > 0 ? "negative" : "default"}
          subtext="columns"
        />
        <MetricCard
          label="Avg Skewness Δ"
          value={avg_skewness_change.toFixed(3)}
          variant={
            avg_skewness_change < -0.1
              ? "positive"
              : avg_skewness_change > 0.1
              ? "negative"
              : "default"
          }
          icon={skewIcon}
        />
        <MetricCard
          label="Avg Outlier Δ"
          value={`${avg_outlier_change.toFixed(1)}%`}
          variant={
            avg_outlier_change < -1
              ? "positive"
              : avg_outlier_change > 1
              ? "negative"
              : "default"
          }
          icon={outlierIcon}
        />
      </div>

      {/* Risky columns row */}
      {risky_columns.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="flex items-center gap-1 text-xs font-medium text-amber-600 dark:text-amber-400">
            <TriangleAlert className="h-3.5 w-3.5" />
            Risky columns:
          </span>
          {risky_columns.map(({ column, reasons }) => (
            <Badge
              key={column}
              variant="outline"
              title={reasons.join("; ")}
              className="border-amber-400 text-amber-700 dark:text-amber-300 text-xs cursor-help"
            >
              {column}
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}
