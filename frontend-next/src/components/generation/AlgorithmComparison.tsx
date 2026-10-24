"use client";

import { useState } from "react";
import { SDV_ALGORITHM_INFO, SYNTHCITY_ALGORITHM_INFO } from "@/lib/constants";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ChevronDown, ChevronRight } from "lucide-react";

/** All models in a tab side by side, so users can pick without opening each one */
export function AlgorithmComparison({ mode }: { mode: "sdv" | "synthcity" }) {
  const [open, setOpen] = useState(false);
  const info = mode === "sdv" ? SDV_ALGORITHM_INFO : SYNTHCITY_ALGORITHM_INFO;
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        Compare all algorithms
      </CollapsibleTrigger>
      <CollapsibleContent>
        <Table className="mt-2">
          <TableHeader>
            <TableRow>
              <TableHead>Algorithm</TableHead>
              <TableHead>What it does</TableHead>
              <TableHead>Best for</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {Object.entries(info).map(([key, v]) => (
              <TableRow key={key}>
                <TableCell className="font-medium">{key}</TableCell>
                <TableCell className="text-xs">{v.desc}</TableCell>
                <TableCell className="text-xs text-muted-foreground">{v.use.replace("Best for: ", "")}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CollapsibleContent>
    </Collapsible>
  );
}
