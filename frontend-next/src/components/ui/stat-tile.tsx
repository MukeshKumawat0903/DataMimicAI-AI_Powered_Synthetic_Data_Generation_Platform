import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { IconTile, type Tone } from "./icon-tile";

/** A single headline number with a label, e.g. "Rows 999" */
export function StatTile({
  label,
  value,
  sub,
  icon,
  tone = "indigo",
  warn,
  className,
}: {
  label: string;
  value: string | number;
  sub?: string;
  icon?: LucideIcon;
  tone?: Tone;
  warn?: boolean;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-xl border bg-card px-4 py-3 transition-shadow hover:shadow-md hover:shadow-indigo-500/5",
        className
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <p className="text-xs font-medium text-muted-foreground">{label}</p>
        {icon && <IconTile icon={icon} tone={warn ? "amber" : tone} size="sm" />}
      </div>
      <p className={cn("mt-1 text-2xl font-bold tabular-nums tracking-tight", warn && "text-amber-600 dark:text-amber-400")}>
        {value}
      </p>
      {sub && <p className="text-xs text-muted-foreground">{sub}</p>}
    </div>
  );
}
