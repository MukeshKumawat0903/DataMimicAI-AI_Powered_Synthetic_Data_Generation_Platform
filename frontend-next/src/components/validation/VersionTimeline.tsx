"use client";

import { useRef, useState } from "react";
import { useWorkspaceStore } from "@/lib/store/workspace";
import { useRestoreVersion } from "@/hooks/api/useGenerate";
import { apiClient } from "@/lib/api/client";
import { downloadBlob, downloadText } from "@/lib/download";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Clock, Download, FileDown, RotateCcw, Save, Upload } from "lucide-react";
import { IconTile } from "@/components/ui/icon-tile";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { toast } from "sonner";
import type { AppError, GenerationRecord, QualityScore } from "@/lib/api/types";

const SCORE_LINES = [
  { key: "fidelity", label: "Fidelity", color: "#2563eb" },
  { key: "privacy", label: "Privacy", color: "#16a34a" },
  { key: "utility", label: "Utility", color: "#f59e0b" },
] as const;

const avgScore = (s: QualityScore) => {
  const xs = [s.fidelity, s.privacy, s.utility].filter((x): x is number => x !== null);
  return xs.reduce((a, b) => a + b, 0) / xs.length;
};

/** Streamlit's rule: converged when the average score moves ≤ 2 points across the last 3 versions */
function hasConverged(scored: GenerationRecord[]) {
  if (scored.length < 3) return false;
  const last = scored.slice(-3).map((h) => avgScore(h.scores!));
  return Math.max(...last) - Math.min(...last) <= 2;
}

function reportMarkdown(history: GenerationRecord[]) {
  return [
    "# Refinement report",
    "",
    "| Version | Model | Rows | Epochs | Fidelity | Privacy | Utility | Time |",
    "|---|---|---|---|---|---|---|---|",
    ...history.map(
      (h) =>
        `| v${h.version} | ${h.algorithm} | ${h.numRows} | ${h.epochs ?? "—"} | ${h.scores?.fidelity ?? "—"} | ${h.scores?.privacy ?? "—"} | ${h.scores?.utility ?? "—"} | ${new Date(h.timestamp).toLocaleString()} |`
    ),
  ].join("\n");
}

function CompareVersions({ history }: { history: GenerationRecord[] }) {
  const [a, setA] = useState(String(history[history.length - 2].version));
  const [b, setB] = useState(String(history[history.length - 1].version));
  const va = history.find((h) => String(h.version) === a)!;
  const vb = history.find((h) => String(h.version) === b)!;
  const delta = (x: number | null | undefined, y: number | null | undefined) =>
    x == null || y == null ? "" : y - x === 0 ? "±0" : `${y - x > 0 ? "+" : ""}${y - x}`;
  const rows: [string, string | number | null | undefined, string | number | null | undefined, string][] = [
    ["Model", va.algorithm, vb.algorithm, ""],
    ["Rows", va.numRows, vb.numRows, delta(va.numRows, vb.numRows)],
    ["Epochs", va.epochs ?? "—", vb.epochs ?? "—", delta(va.epochs, vb.epochs)],
    ...SCORE_LINES.map(
      (l) =>
        [l.label, va.scores?.[l.key] ?? "—", vb.scores?.[l.key] ?? "—", delta(va.scores?.[l.key], vb.scores?.[l.key])] as [
          string, string | number, string | number, string,
        ]
    ),
  ];

  const picker = (id: string, value: string, set: (v: string) => void) => (
    <Select value={value} onValueChange={(v) => set(v as string)}>
      <SelectTrigger id={id} className="w-24">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {history.map((h) => (
          <SelectItem key={h.version} value={String(h.version)}>
            v{h.version}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );

  return (
    <div className="space-y-2 rounded-lg border p-3">
      <div className="flex flex-wrap items-end gap-3">
        <p className="text-sm font-medium mr-2">Compare versions</p>
        <div className="space-y-1">
          <Label htmlFor="cmp-a">From</Label>
          {picker("cmp-a", a, setA)}
        </div>
        <div className="space-y-1">
          <Label htmlFor="cmp-b">To</Label>
          {picker("cmp-b", b, setB)}
        </div>
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead />
            <TableHead>v{va.version}</TableHead>
            <TableHead>v{vb.version}</TableHead>
            <TableHead>Change</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map(([label, x, y, d]) => (
            <TableRow key={label}>
              <TableCell className="text-muted-foreground">{label}</TableCell>
              <TableCell>{x}</TableCell>
              <TableCell>{y}</TableCell>
              <TableCell className={d.startsWith("+") ? "text-green-600" : d.startsWith("-") ? "text-destructive" : ""}>{d}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

export function VersionTimeline() {
  const fileId = useWorkspaceStore((s) => s.fileId);
  const history = useWorkspaceStore((s) => s.generationHistory);
  const currentVersionId = useWorkspaceStore((s) => s.currentVersionId);
  const { generatorConfig, selectedAlgorithm, restoreSession } = useWorkspaceStore();
  const restore = useRestoreVersion();
  const fileInput = useRef<HTMLInputElement>(null);
  const scored = history.filter((h) => h.scores);
  const latestVersionId = history[history.length - 1]?.versionId ?? null;
  const current = currentVersionId ?? latestVersionId;

  async function downloadVersion(h: GenerationRecord) {
    try {
      const res = await apiClient.get(`/eda/download`, { params: { file_id: `syn_${fileId}__${h.versionId}` }, responseType: "blob" });
      downloadBlob(res.data as Blob, `synthetic_v${h.version}.csv`);
    } catch (e) {
      toast.error("Download failed", { description: (e as Partial<AppError>).message });
    }
  }

  function saveSession() {
    const session = { fileId, savedAt: new Date().toISOString(), history, generatorConfig, selectedAlgorithm };
    downloadText(JSON.stringify(session, null, 2), "datamimic_session.json", "application/json");
  }

  async function loadSession(file: File) {
    try {
      const s = JSON.parse(await file.text());
      if (!Array.isArray(s.history)) throw new Error("Not a session file");
      restoreSession({ history: s.history, config: s.generatorConfig, algorithm: s.selectedAlgorithm });
      toast.success(`Loaded ${s.history.length} version(s)`, {
        description: s.fileId && s.fileId !== fileId ? "Saved for a different dataset: restore won't find its files." : undefined,
      });
    } catch (e) {
      toast.error("Couldn't load session", { description: (e as Error).message });
    }
  }

  return (
    <Card>
      <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2 py-3">
        <div className="flex items-center gap-3">
          <IconTile icon={Clock} tone="indigo" />
          <CardTitle className="text-base">Generation History</CardTitle>
        </div>
        <div className="flex flex-wrap gap-1">
          {history.length > 0 && (
            <>
              <Button size="sm" variant="ghost" className="gap-1" onClick={() => downloadText(reportMarkdown(history), "refinement_report.md", "text/markdown")}>
                <FileDown className="h-3.5 w-3.5" /> Report
              </Button>
              <Button size="sm" variant="ghost" className="gap-1" onClick={saveSession}>
                <Save className="h-3.5 w-3.5" /> Save session
              </Button>
            </>
          )}
          <Button size="sm" variant="ghost" className="gap-1" onClick={() => fileInput.current?.click()}>
            <Upload className="h-3.5 w-3.5" /> Load session
          </Button>
          <input
            ref={fileInput}
            type="file"
            accept="application/json"
            className="hidden"
            aria-label="Load session file"
            onChange={(e) => e.target.files?.[0] && loadSession(e.target.files[0])}
          />
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {history.length === 0 ? (
          <p className="text-sm text-muted-foreground">No generations yet. Use the Refinement panel to generate data.</p>
        ) : (
          <>
            {hasConverged(scored) && (
              <p className="rounded border border-blue-200 bg-blue-50 px-3 py-2 text-sm text-blue-700">
                Scores have converged (within 2 points over the last 3 versions). More regeneration with the same
                settings is unlikely to help; try another model or different parameters.
              </p>
            )}
            {scored.length >= 2 && (
              <div className="h-48" aria-label="Quality score progression">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={scored.map((h) => ({ version: `v${h.version}`, ...h.scores }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="version" fontSize={12} />
                    <YAxis domain={[0, 100]} fontSize={12} />
                    <Tooltip />
                    <Legend />
                    {SCORE_LINES.map((l) => (
                      <Line key={l.key} dataKey={l.key} name={l.label} stroke={l.color} dot connectNulls />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
            {history.length >= 2 && <CompareVersions history={history} />}
            <div className="space-y-2">
              {[...history].reverse().map((entry) => {
                const isCurrent = !!entry.versionId && entry.versionId === current;
                return (
                  <div key={entry.version} className="flex flex-wrap items-center justify-between gap-2 rounded border px-3 py-2 text-sm">
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-muted-foreground font-mono">v{entry.version}</span>
                      <div>
                        <span className="font-medium">{entry.algorithm}</span>
                        <span className="mx-1 text-muted-foreground">·</span>
                        <span className="text-muted-foreground">{entry.numRows.toLocaleString()} rows</span>
                        {entry.epochs && (
                          <>
                            <span className="mx-1 text-muted-foreground">·</span>
                            <span className="text-muted-foreground">{entry.epochs} epochs</span>
                          </>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {entry.scores && (
                        <span className="text-xs text-muted-foreground">
                          F {entry.scores.fidelity} · P {entry.scores.privacy ?? "n/a"} · U {entry.scores.utility}
                        </span>
                      )}
                      <span className="text-xs text-muted-foreground">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                      {entry.versionId && (
                        <Button size="sm" variant="ghost" className="h-7 px-2" aria-label={`Download v${entry.version}`} onClick={() => downloadVersion(entry)}>
                          <Download className="h-3.5 w-3.5" />
                        </Button>
                      )}
                      {isCurrent ? (
                        <Badge variant="secondary" className="text-xs">current</Badge>
                      ) : (
                        entry.versionId &&
                        fileId && (
                          <Button size="sm" variant="outline" className="h-7 gap-1 px-2 text-xs" disabled={restore.isPending}
                            onClick={() =>
                              restore.mutate(
                                { fileId, versionId: entry.versionId! },
                                {
                                  onSuccess: () => toast.success(`v${entry.version} is the current dataset again`),
                                  onError: (e) => toast.error("Restore failed", { description: (e as Partial<AppError>).message }),
                                }
                              )
                            }>
                            <RotateCcw className="h-3 w-3" /> Restore
                          </Button>
                        )
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
