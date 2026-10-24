import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

/** Colour families used to tell sections apart at a glance */
export const TONES = {
  brand: "bg-brand-gradient text-white shadow-sm shadow-indigo-500/30",
  violet: "bg-violet-100 text-violet-600 dark:bg-violet-500/15 dark:text-violet-300",
  indigo: "bg-indigo-100 text-indigo-600 dark:bg-indigo-500/15 dark:text-indigo-300",
  sky: "bg-sky-100 text-sky-600 dark:bg-sky-500/15 dark:text-sky-300",
  cyan: "bg-cyan-100 text-cyan-700 dark:bg-cyan-500/15 dark:text-cyan-300",
  emerald: "bg-emerald-100 text-emerald-600 dark:bg-emerald-500/15 dark:text-emerald-300",
  amber: "bg-amber-100 text-amber-600 dark:bg-amber-500/15 dark:text-amber-300",
  orange: "bg-orange-100 text-orange-600 dark:bg-orange-500/15 dark:text-orange-300",
  rose: "bg-rose-100 text-rose-600 dark:bg-rose-500/15 dark:text-rose-300",
  pink: "bg-fuchsia-100 text-fuchsia-600 dark:bg-fuchsia-500/15 dark:text-fuchsia-300",
  slate: "bg-slate-100 text-slate-600 dark:bg-slate-500/20 dark:text-slate-300",
} as const;

export type Tone = keyof typeof TONES;

const SIZES = {
  sm: "h-7 w-7 rounded-lg [&>svg]:h-3.5 [&>svg]:w-3.5",
  md: "h-9 w-9 rounded-xl [&>svg]:h-[18px] [&>svg]:w-[18px]",
  lg: "h-12 w-12 rounded-2xl [&>svg]:h-6 [&>svg]:w-6",
} as const;

export function IconTile({
  icon: Icon,
  tone = "brand",
  size = "md",
  className,
}: {
  icon: LucideIcon;
  tone?: Tone;
  size?: keyof typeof SIZES;
  className?: string;
}) {
  return (
    <span aria-hidden className={cn("inline-flex shrink-0 items-center justify-center", SIZES[size], TONES[tone], className)}>
      <Icon />
    </span>
  );
}
