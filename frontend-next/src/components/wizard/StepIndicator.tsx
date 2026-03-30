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
      className="hidden md:flex items-center justify-center gap-0 border-b bg-background px-4 py-2 text-xs"
    >
      {STEPS.map((step, idx) => {
        const isActive = idx === currentIdx;
        const isDone = idx < currentIdx;
        const accessible = isAccessible(idx);

        const label = (
          <span
            className={cn(
              "flex items-center gap-1.5 font-medium",
              isActive && "text-primary",
              isDone && "text-green-600 dark:text-green-400",
              !isDone && !isActive && "text-muted-foreground"
            )}
          >
            {/* Step marker */}
            <span
              className={cn(
                "flex h-5 w-5 shrink-0 items-center justify-center rounded-full border text-[10px] font-bold",
                isActive && "border-primary bg-primary text-primary-foreground",
                isDone && "border-green-500 bg-green-500 text-white",
                !isDone && !isActive && "border-muted-foreground/40 text-muted-foreground"
              )}
            >
              {isDone ? <Check className="h-3 w-3" strokeWidth={3} /> : idx + 1}
            </span>
            {step.label}
          </span>
        );

        return (
          <div key={step.href} className="flex items-center">
            {accessible && !isActive ? (
              <Link href={step.href} className="rounded px-3 py-1 hover:bg-accent transition-colors">
                {label}
              </Link>
            ) : (
              <span className={cn("px-3 py-1", isActive && "cursor-default")}>{label}</span>
            )}
            {idx < STEPS.length - 1 && (
              <span className="text-muted-foreground/40 select-none mx-0.5">→</span>
            )}
          </div>
        );
      })}
    </nav>
  );
}
