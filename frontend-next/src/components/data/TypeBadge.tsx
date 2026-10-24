import { cn } from "@/lib/utils";

/** Column type → colour, so numeric, text and date columns are easy to tell apart */
function typeColour(type: string) {
  const t = type.toLowerCase();
  if (/date|time|period/.test(t)) return "bg-cyan-50 text-cyan-700 ring-cyan-600/20 dark:bg-cyan-500/10 dark:text-cyan-300";
  if (/bool/.test(t)) return "bg-fuchsia-50 text-fuchsia-700 ring-fuchsia-600/20 dark:bg-fuchsia-500/10 dark:text-fuchsia-300";
  if (/float|double|decimal|continuous/.test(t)) return "bg-violet-50 text-violet-700 ring-violet-600/20 dark:bg-violet-500/10 dark:text-violet-300";
  if (/int|numeric|number/.test(t)) return "bg-sky-50 text-sky-700 ring-sky-600/20 dark:bg-sky-500/10 dark:text-sky-300";
  if (/object|str|text|categor/.test(t)) return "bg-amber-50 text-amber-700 ring-amber-600/20 dark:bg-amber-500/10 dark:text-amber-300";
  return "bg-slate-50 text-slate-700 ring-slate-500/20 dark:bg-slate-500/10 dark:text-slate-300";
}

export function TypeBadge({ type, className }: { type: string; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex h-5 items-center rounded-full px-2 font-mono text-[11px] font-medium ring-1 ring-inset",
        typeColour(type),
        className
      )}
    >
      {type}
    </span>
  );
}
