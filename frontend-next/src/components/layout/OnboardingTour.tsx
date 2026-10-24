"use client";

import { useEffect, useState } from "react";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { BarChart2, ShieldCheck, Sparkles, Upload, Wand2 } from "lucide-react";

const SEEN_KEY = "datamimic-tour-seen";

const STEPS = [
  { icon: Sparkles, title: "Welcome to DataMimicAI",
    text: "Turn a real dataset into a privacy-safe synthetic copy in four steps. This short tour shows where everything is." },
  { icon: Upload, title: "1. Upload",
    text: "Upload a CSV or load a demo dataset. You get a data overview, a quality score and a preview table." },
  { icon: BarChart2, title: "2. Explore",
    text: "Profile, clean and analyse your data: correlations, outliers, privacy risks, time series. AI Assistance can diagnose issues and propose fixes that you approve." },
  { icon: Wand2, title: "3. Generate",
    text: "Pick a model (SDV or SynthCity), choose a preset, and generate. Accepted feature suggestions are applied automatically." },
  { icon: ShieldCheck, title: "4. Validate",
    text: "Check fidelity, privacy and utility, compare real and synthetic data in charts, and refine until you're happy. Every version can be restored." },
];

/** Opens once on the first visit; reopen any time from the sidebar */
export function OnboardingTour() {
  const open = useWorkspaceStore((s) => s.tourOpen);
  const setOpen = useWorkspaceStore((s) => s.setTourOpen);
  const [i, setI] = useState(0);

  useEffect(() => {
    try {
      if (!localStorage.getItem(SEEN_KEY)) setOpen(true);
    } catch {
      // storage unavailable: don't auto-open
    }
  }, [setOpen]);

  function close() {
    try {
      localStorage.setItem(SEEN_KEY, "1");
    } catch {
      // ignore
    }
    setOpen(false);
    setI(0);
  }

  const step = STEPS[i];
  const Icon = step.icon;
  return (
    <Dialog open={open} onOpenChange={(o) => (o ? setOpen(true) : close())}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            <span className="bg-brand-gradient flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-white shadow-md shadow-indigo-500/30">
              <Icon className="h-5 w-5" />
            </span>
            {step.title}
          </DialogTitle>
        </DialogHeader>
        <p className="text-sm text-muted-foreground">{step.text}</p>
        <div className="flex items-center justify-between pt-2">
          <span className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="flex gap-1" aria-hidden>
              {STEPS.map((_, d) => (
                <span key={d} className={`h-1.5 rounded-full transition-all ${d === i ? "w-5 bg-primary" : "w-1.5 bg-muted-foreground/30"}`} />
              ))}
            </span>
            {i + 1} / {STEPS.length}
          </span>
          <div className="flex gap-2">
            {i > 0 && (
              <Button variant="outline" size="sm" onClick={() => setI(i - 1)}>
                Back
              </Button>
            )}
            {i < STEPS.length - 1 ? (
              <Button size="sm" onClick={() => setI(i + 1)}>
                Next
              </Button>
            ) : (
              <Button size="sm" onClick={close}>
                Get started
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
