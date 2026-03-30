"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  Upload,
  BarChart2,
  Wand2,
  ShieldCheck,
  Database,
} from "lucide-react";

const steps = [
  { label: "1. Upload Data", href: "/upload", icon: Upload },
  { label: "2. Explore (EDA)", href: "/explore", icon: BarChart2 },
  { label: "3. Generate", href: "/generate", icon: Wand2 },
  { label: "4. Validate", href: "/validate", icon: ShieldCheck },
];

export function AppSidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden md:flex w-60 flex-col border-r bg-card px-4 py-6 gap-6">
      {/* Logo / brand */}
      <div className="flex items-center gap-2 px-2">
        <Database className="h-6 w-6 text-primary" />
        <span className="font-semibold text-lg tracking-tight">DataMimicAI</span>
      </div>

      {/* Step navigation */}
      <nav className="flex flex-col gap-1">
        {steps.map(({ label, href, icon: Icon }) => {
          const active = pathname === href || pathname.startsWith(href + "/");
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                active
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
              )}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
