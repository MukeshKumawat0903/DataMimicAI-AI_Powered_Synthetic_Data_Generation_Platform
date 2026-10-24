"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { cn } from "@/lib/utils";
import { Check } from "lucide-react";

const STEPS = [
  { label: "Upload", href: "/upload" },
  { label: "Explore", href: "/explore" },
  { label: "Generate", href: "/generate" },
  { label: "Validate", href: "/validate" },
] as const;

/**
 * Horizontal wizard progress bar displayed at the top of every page.
 * Steps that appear before the current page are marked as completed.
 * Restricted steps (no fileId) render as non-clickable spans.
 */
export function StepIndicator() {
  const pathname = usePathname();
  const fileId = useWorkspaceStore((s) => s.fileId);
  const generatedFileId = useWorkspaceStore((s) => s.generatedFileId);

  const currentIdx = STEPS.findIndex(
    (s) => pathname === s.href || pathname.startsWith(s.href + "/")
  );

  function isAccessible(idx: number): boolean {
    if (idx === 0) return true; // upload always accessible
    if (!fileId) return false; // nothing accessible without a file
    if (idx <= 2) return true; // explore + generate accessible once file loaded
    return !!generatedFileId; // validate only after generation
  }

  return (
    <nav
      aria-label="Wizard progress"
      className="hidden md:flex items-center justify-center border-b bg-background/80 px-4 py-2.5 text-sm backdrop-blur supports-[backdrop-filter]:bg-background/70"
    >
      {STEPS.map((step, idx) => {
        const isActive = idx === currentIdx;
        const isDone = idx < currentIdx;
        const accessible = isAccessible(idx);

        const label = (
          <span
            className={cn(
              "flex items-center gap-2 font-medium",
              isActive && "text-foreground",
              isDone && "text-emerald-600 dark:text-emerald-400",
              !isDone && !isActive && "text-muted-foreground"
            )}
          >
            {/* Step marker */}
            <span
              className={cn(
                "flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-bold transition-all duration-300",
                isActive && "bg-brand-gradient text-white shadow-md shadow-indigo-500/30 ring-4 ring-primary/15",
                isDone && "bg-emerald-500 text-white",
                !isDone && !isActive && "border border-border bg-background text-muted-foreground"
              )}
            >
              {isDone ? <Check className="h-3.5 w-3.5" strokeWidth={3} /> : idx + 1}
            </span>
            {step.label}
          </span>
        );

        return (
          <div key={step.href} className="flex items-center">
            {accessible && !isActive ? (
              <Link href={step.href} className="rounded-full px-2.5 py-1 transition-colors hover:bg-accent">
                {label}
              </Link>
            ) : (
              <span className={cn("px-2.5 py-1", isActive && "cursor-default")}>{label}</span>
            )}
            {idx < STEPS.length - 1 && (
              <span
                aria-hidden
                className={cn(
                  "mx-1 h-0.5 w-8 rounded-full lg:w-14",
                  idx < currentIdx ? "bg-emerald-400" : "bg-border"
                )}
              />
            )}
          </div>
        );
      })}
    </nav>
  );
}
