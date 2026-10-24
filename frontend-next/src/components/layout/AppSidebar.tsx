"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { cn } from "@/lib/utils";
import { apiClient } from "@/lib/api/client";
import { downloadBlob } from "@/lib/download";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { ThemeToggle } from "./ThemeToggle";
import { OnboardingTour } from "./OnboardingTour";
import { Button } from "@/components/ui/button";
import { Upload, BarChart2, Wand2, ShieldCheck, Check, Lock, Download, ArrowRight, Compass, RotateCcw } from "lucide-react";
import { toast } from "sonner";
import type { AppError } from "@/lib/api/types";

const steps = [
  { label: "Upload Data", hint: "CSV file or demo", href: "/upload", icon: Upload, needs: null },
  { label: "Explore (EDA)", hint: "Profile, clean, get AI help", href: "/explore", icon: BarChart2, needs: "Upload a dataset first" },
  { label: "Generate", hint: "Train a synthesizer", href: "/generate", icon: Wand2, needs: "Upload a dataset first" },
  { label: "Validate", hint: "Quality, privacy, refine", href: "/validate", icon: ShieldCheck, needs: "Generate synthetic data first" },
];

// Ghost buttons on the dark sidebar
const sideButton = "justify-start gap-2 text-indigo-50/90 hover:bg-white/15 hover:text-white aria-expanded:bg-white/15 aria-expanded:text-white";

function ApiStatus() {
  const { data, isLoading } = useQuery({
    queryKey: ["api-health"],
    queryFn: () => apiClient.get(`/health`, { timeout: 5000 }).then(() => true),
    refetchInterval: 30_000,
    retry: false,
  });
  const online = data === true;
  return (
    <div className="flex items-center gap-2 px-3 pt-1 text-xs text-indigo-100/85" aria-live="polite">
      <span className="relative flex h-2 w-2">
        {online && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />}
        <span className={cn("relative inline-flex h-2 w-2 rounded-full", isLoading ? "bg-indigo-300/60" : online ? "bg-emerald-400" : "bg-rose-400")} />
      </span>
      {isLoading ? "Checking API…" : online ? "API online" : "API offline"}
    </div>
  );
}

export function AppSidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const qc = useQueryClient();
  const fileId = useWorkspaceStore((s) => s.fileId);
  const generatedFileId = useWorkspaceStore((s) => s.generatedFileId);
  const { reset, setTourOpen } = useWorkspaceStore();
  const [confirmReset, setConfirmReset] = useState(false);

  const currentIdx = steps.findIndex((s) => pathname === s.href || pathname.startsWith(s.href + "/"));
  const accessible = (idx: number) => idx === 0 || (!!fileId && (idx <= 2 || !!generatedFileId));
  const progress = Math.max(currentIdx + 1, 1);

  async function download(file: string, name: string) {
    try {
      const res = await apiClient.get(`/eda/download`, { params: { file_id: file }, responseType: "blob" });
      downloadBlob(res.data as Blob, name);
    } catch (e) {
      toast.error("Download failed", { description: (e as Partial<AppError>).message });
    }
  }

  function startOver() {
    reset();
    qc.clear();
    setConfirmReset(false);
    router.push("/upload");
  }

  return (
    <aside className="bg-sidebar-brand hidden md:flex w-64 shrink-0 flex-col gap-6 px-4 py-6 text-white">
      <div className="flex items-center gap-3 px-2">
        <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/15 ring-1 ring-white/20 shadow-sm shadow-indigo-950/30">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/logo.png" alt="" className="h-7 w-7" />
        </span>
        <div className="leading-tight">
          <p className="text-lg font-semibold tracking-tight">DataMimicAI</p>
          <p className="text-[11px] text-indigo-100/90">Synthetic data studio</p>
        </div>
      </div>

      {/* Step navigation: completed steps get a check, unavailable ones are locked */}
      <nav className="flex flex-col gap-1">
        <p className="px-3 pb-1 text-[11px] font-semibold uppercase tracking-wider text-indigo-100/80">Workflow</p>
        {steps.map(({ label, hint, href, icon: Icon, needs }, idx) => {
          const active = idx === currentIdx;
          const done = idx < currentIdx && accessible(idx);
          if (!accessible(idx)) {
            return (
              <span key={href} title={needs ?? undefined} aria-disabled="true"
                className="flex items-center gap-3 rounded-xl px-3 py-2 text-sm font-medium text-indigo-100/55 cursor-not-allowed">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white/[0.08]">
                  <Lock className="h-3.5 w-3.5" />
                </span>
                <span className="leading-tight">
                  {label}
                  <span className="block text-[11px] font-normal">{needs}</span>
                </span>
              </span>
            );
          }
          return (
            <Link key={href} href={href}
              className={cn(
                "group flex items-center gap-3 rounded-xl px-3 py-2 text-sm font-medium transition-all duration-200",
                active ? "bg-white/[0.16] text-white ring-1 ring-white/25 shadow-lg shadow-indigo-950/30" : "text-indigo-50/90 hover:bg-white/10 hover:text-white"
              )}>
              <span className={cn(
                "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg transition-colors",
                active ? "bg-brand-gradient shadow-md shadow-indigo-950/40 ring-1 ring-white/30" : done ? "bg-emerald-400/20 text-emerald-200" : "bg-white/[0.12] group-hover:bg-white/20"
              )}>
                {done ? <Check className="h-4 w-4" strokeWidth={2.5} /> : <Icon className="h-4 w-4" />}
              </span>
              <span className="leading-tight">
                {label}
                <span className={cn("block text-[11px] font-normal", active ? "text-indigo-50/90" : "text-indigo-100/80")}>{hint}</span>
              </span>
            </Link>
          );
        })}
      </nav>

      <div className="mx-1 rounded-xl bg-white/10 px-3 py-3 ring-1 ring-white/15">
        <div className="flex items-baseline justify-between text-xs">
          <span className="font-medium text-indigo-100/90">Progress</span>
          <span className="tabular-nums text-indigo-100/85">Step {progress} of {steps.length}</span>
        </div>
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/15">
          <div className="bg-brand-gradient h-full rounded-full transition-all duration-500" style={{ width: `${(progress / steps.length) * 100}%` }} />
        </div>
      </div>

      {fileId && (
        <div className="flex flex-col gap-1">
          <p className="px-3 pb-1 text-[11px] font-semibold uppercase tracking-wider text-indigo-100/80">Quick actions</p>
          <Button variant="ghost" size="sm" className={sideButton} onClick={() => download(fileId, "dataset.csv")}>
            <Download className="h-4 w-4" /> Download dataset
          </Button>
          {generatedFileId && (
            <>
              <Button variant="ghost" size="sm" className={sideButton} onClick={() => download(`syn_${fileId}`, "synthetic_data.csv")}>
                <Download className="h-4 w-4" /> Download synthetic
              </Button>
              {pathname !== "/validate" && (
                <Button variant="ghost" size="sm" className={sideButton} onClick={() => router.push("/validate")}>
                  <ArrowRight className="h-4 w-4" /> Jump to Validate
                </Button>
              )}
            </>
          )}
        </div>
      )}

      <div className="mt-auto flex flex-col gap-1 border-t border-white/15 pt-4">
        <Button variant="ghost" size="sm" className={sideButton} onClick={() => setTourOpen(true)}>
          <Compass className="h-4 w-4" /> Take the tour
        </Button>
        <ThemeToggle className={sideButton} />
        {fileId &&
          (confirmReset ? (
            <Button variant="destructive" size="sm" className="justify-start gap-2 bg-rose-500/20 text-rose-100 hover:bg-rose-500/30" onClick={startOver}>
              <RotateCcw className="h-4 w-4" /> Confirm start over
            </Button>
          ) : (
            <Button variant="ghost" size="sm" className={sideButton} onClick={() => setConfirmReset(true)}>
              <RotateCcw className="h-4 w-4" /> Start over
            </Button>
          ))}
        <ApiStatus />
      </div>
      <OnboardingTour />
    </aside>
  );
}
