import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { IconTile } from "@/components/ui/icon-tile";

/** Title block shared by the four wizard pages */
export function PageHeader({
  icon,
  step,
  title,
  description,
  actions,
}: {
  icon: LucideIcon;
  /** e.g. "Step 2 of 4" */
  step?: string;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div className="flex items-start gap-4">
        <IconTile icon={icon} size="lg" />
        <div>
          {step && <p className="text-xs font-semibold uppercase tracking-wider text-primary">{step}</p>}
          <h1 className="text-2xl font-bold tracking-tight sm:text-[1.75rem]">{title}</h1>
          <p className="mt-1 max-w-xl text-sm text-muted-foreground">{description}</p>
        </div>
      </div>
      {actions && <div className="flex flex-wrap items-center gap-1">{actions}</div>}
    </div>
  );
}
