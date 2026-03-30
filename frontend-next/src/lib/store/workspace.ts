import { create } from "zustand";
import type { ColumnInfo, KPISummary, Algorithm, FeedbackItem, GeneratorConfig, QualityScore } from "@/lib/api/types";

interface WorkspaceState {
  // ── Dataset identity ──────────────────────────────────────────────────────
  fileId: string | null;
  columns: string[];
  columnInfo: ColumnInfo[];
  rowCount: number;

  // ── Generated data ────────────────────────────────────────────────────────
  generatedFileId: string | null;

  // ── KPI / smart preview ───────────────────────────────────────────────────
  kpiSummary: KPISummary | null;

  // ── UI state ──────────────────────────────────────────────────────────────
  demoAlgorithm: string;

  // ── Generation ────────────────────────────────────────────────────────────
  selectedAlgorithm: Algorithm;
  generatorConfig: GeneratorConfig;
  pendingFeedback: FeedbackItem[];

  // ── Quality scores ────────────────────────────────────────────────────────
  qualityScores: QualityScore | null;

  // ── Actions ───────────────────────────────────────────────────────────────
  setUploadResult: (result: {
    fileId: string;
    columns: string[];
    rowCount: number;
  }) => void;
  setColumnInfo: (info: ColumnInfo[]) => void;
  setGeneratedFileId: (id: string | null) => void;
  setKpiSummary: (summary: KPISummary) => void;
  setDemoAlgorithm: (algo: string) => void;
  setSelectedAlgorithm: (algo: Algorithm) => void;
  setGeneratorConfig: (cfg: Partial<GeneratorConfig>) => void;
  addFeedback: (item: FeedbackItem) => void;
  clearFeedback: () => void;
  setQualityScores: (scores: QualityScore) => void;
  reset: () => void;
}

const initialState = {
  fileId: null,
  columns: [],
  columnInfo: [],
  rowCount: 0,
  generatedFileId: null,
  kpiSummary: null,
  demoAlgorithm: "CTGAN",
  selectedAlgorithm: "CTGAN" as Algorithm,
  generatorConfig: { algorithm: "CTGAN" as Algorithm, num_rows: 1000 },
  pendingFeedback: [] as FeedbackItem[],
  qualityScores: null,
};

export const useWorkspaceStore = create<WorkspaceState>((set) => ({
  ...initialState,

  setUploadResult: ({ fileId, columns, rowCount }) =>
    set({ fileId, columns, rowCount, generatedFileId: null, kpiSummary: null }),

  setColumnInfo: (info) => set({ columnInfo: info }),

  setGeneratedFileId: (id) => set({ generatedFileId: id }),

  setKpiSummary: (summary) => set({ kpiSummary: summary }),

  setDemoAlgorithm: (algo) => set({ demoAlgorithm: algo }),

  setSelectedAlgorithm: (algo) =>
    set((s) => ({
      selectedAlgorithm: algo,
      generatorConfig: { ...s.generatorConfig, algorithm: algo },
    })),

  setGeneratorConfig: (cfg) =>
    set((s) => ({ generatorConfig: { ...s.generatorConfig, ...cfg } })),

  addFeedback: (item) =>
    set((s) => ({ pendingFeedback: [...s.pendingFeedback, item] })),

  clearFeedback: () => set({ pendingFeedback: [] }),

  setQualityScores: (scores) => set({ qualityScores: scores }),

  reset: () => set(initialState),
}));
