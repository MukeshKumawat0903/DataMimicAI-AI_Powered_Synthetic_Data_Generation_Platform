import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type {
  ColumnInfo,
  KPISummary,
  Algorithm,
  DiagnosticsBuildResponse,
  DiagnosticsInterpretation,
  FeedbackItem,
  GenerationRecord,
  GeneratorConfig,
  PlanExecution,
  PlanReview,
  QualityScore,
  TransformDecision,
  TransformationPlan,
} from "@/lib/api/types";

const decisionKey = (d: TransformDecision) => `${d.column}|${d.category}|${d.transformation}`;

export type ExploreMode = "analysis" | "ai";

/** Action Planner progress for the current dataset (diagnose → plan → review → execute) */
export interface PlannerProgress {
  built?: DiagnosticsBuildResponse;
  interpretation?: DiagnosticsInterpretation;
  plans?: TransformationPlan[];
  reviews: Record<string, PlanReview>;
  executions: Record<string, PlanExecution>;
}

const emptyPlanner = (): PlannerProgress => ({ reviews: {}, executions: {} });

interface WorkspaceState {
  // ── Dataset identity ──────────────────────────────────────────────────────
  fileId: string | null;
  columns: string[];
  columnInfo: ColumnInfo[];
  rowCount: number;

  // ── Generated data ────────────────────────────────────────────────────────
  generatedFileId: string | null;
  /** Every generation in this session, oldest first (each one replaces the synthetic file) */
  generationHistory: GenerationRecord[];
  /** Version currently in the synthetic file (latest, unless an older one was restored) */
  currentVersionId: string | null;

  // ── KPI / smart preview ───────────────────────────────────────────────────
  kpiSummary: KPISummary | null;

  // ── UI state ──────────────────────────────────────────────────────────────
  demoAlgorithm: string;
  exploreMode: ExploreMode;
  /** Onboarding tour dialog (not persisted) */
  tourOpen: boolean;

  // ── AI Assistance ─────────────────────────────────────────────────────────
  planner: PlannerProgress;

  // ── Generation ────────────────────────────────────────────────────────────
  selectedAlgorithm: Algorithm;
  generatorConfig: GeneratorConfig;
  /** Transformations applied by the backend right before generation */
  pendingFeedback: FeedbackItem[];

  // ── Feature engineering ───────────────────────────────────────────────────
  /** Suggestions the user accepted in Explore → Feature Suggestions */
  acceptedDecisions: TransformDecision[];
  /** Columns changed by the last cleaning action (highlighted in the dataset view) */
  lastChangedColumns: string[];

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
  addGeneration: (record: Omit<GenerationRecord, "version">) => void;
  /** Attach quality scores to the latest generation once they're computed */
  setLatestGenerationScores: (scores: QualityScore) => void;
  setCurrentVersionId: (id: string | null) => void;
  /** Replace the session's history and settings (loading a saved session) */
  restoreSession: (s: { history: GenerationRecord[]; config: GeneratorConfig; algorithm: Algorithm }) => void;
  setKpiSummary: (summary: KPISummary | null) => void;
  setDemoAlgorithm: (algo: string) => void;
  setExploreMode: (mode: ExploreMode) => void;
  setTourOpen: (open: boolean) => void;
  updatePlanner: (update: Partial<PlannerProgress> | ((p: PlannerProgress) => Partial<PlannerProgress>)) => void;
  resetPlanner: () => void;
  setSelectedAlgorithm: (algo: Algorithm) => void;
  setGeneratorConfig: (cfg: Partial<GeneratorConfig>) => void;
  addFeedback: (item: FeedbackItem) => void;
  setFeedback: (items: FeedbackItem[]) => void;
  clearFeedback: () => void;
  toggleDecision: (decision: TransformDecision) => void;
  setDecisions: (decisions: TransformDecision[]) => void;
  setLastChangedColumns: (columns: string[]) => void;
  setQualityScores: (scores: QualityScore) => void;
  reset: () => void;
}

const initialState = {
  fileId: null,
  columns: [],
  columnInfo: [],
  rowCount: 0,
  generatedFileId: null,
  generationHistory: [] as GenerationRecord[],
  currentVersionId: null as string | null,
  kpiSummary: null,
  demoAlgorithm: "CTGAN",
  exploreMode: "analysis" as ExploreMode,
  tourOpen: false,
  planner: emptyPlanner(),
  selectedAlgorithm: "CTGAN" as Algorithm,
  generatorConfig: { algorithm: "CTGAN" as Algorithm, num_rows: 1000 },
  pendingFeedback: [] as FeedbackItem[],
  acceptedDecisions: [] as TransformDecision[],
  lastChangedColumns: [] as string[],
  qualityScores: null,
};

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set) => ({
  ...initialState,

  setUploadResult: ({ fileId, columns, rowCount }) =>
    set({
      fileId,
      columns,
      rowCount,
      generatedFileId: null,
      generationHistory: [],
      currentVersionId: null,
      kpiSummary: null,
      planner: emptyPlanner(),
      acceptedDecisions: [],
      pendingFeedback: [],
    }),

  setColumnInfo: (info) => set({ columnInfo: info }),

  setGeneratedFileId: (id) => set({ generatedFileId: id }),

  addGeneration: (record) =>
    set((s) => ({
      generationHistory: [...s.generationHistory, { ...record, version: s.generationHistory.length + 1 }],
      currentVersionId: record.versionId ?? null,
    })),

  setCurrentVersionId: (id) => set({ currentVersionId: id }),

  restoreSession: ({ history, config, algorithm }) =>
    set({
      generationHistory: history,
      generatorConfig: config,
      selectedAlgorithm: algorithm,
      currentVersionId: history[history.length - 1]?.versionId ?? null,
    }),

  setLatestGenerationScores: (scores) =>
    set((s) => {
      const history = [...s.generationHistory];
      const last = history[history.length - 1];
      if (!last || last.scores) return {};
      history[history.length - 1] = { ...last, scores };
      return { generationHistory: history };
    }),

  setKpiSummary: (summary) => set({ kpiSummary: summary }),

  setDemoAlgorithm: (algo) => set({ demoAlgorithm: algo }),

  setExploreMode: (mode) => set({ exploreMode: mode }),

  setTourOpen: (open) => set({ tourOpen: open }),

  updatePlanner: (update) =>
    set((s) => ({
      planner: { ...s.planner, ...(typeof update === "function" ? update(s.planner) : update) },
    })),

  resetPlanner: () => set({ planner: emptyPlanner() }),

  setSelectedAlgorithm: (algo) =>
    set((s) => ({
      selectedAlgorithm: algo,
      generatorConfig: { ...s.generatorConfig, algorithm: algo },
    })),

  setGeneratorConfig: (cfg) =>
    set((s) => ({ generatorConfig: { ...s.generatorConfig, ...cfg } })),

  addFeedback: (item) =>
    set((s) => ({ pendingFeedback: [...s.pendingFeedback, item] })),

  setFeedback: (items) => set({ pendingFeedback: items }),

  clearFeedback: () => set({ pendingFeedback: [] }),

  toggleDecision: (decision) =>
    set((s) => {
      const key = decisionKey(decision);
      const exists = s.acceptedDecisions.some((d) => decisionKey(d) === key);
      return {
        acceptedDecisions: exists
          ? s.acceptedDecisions.filter((d) => decisionKey(d) !== key)
          : [...s.acceptedDecisions, decision],
      };
    }),

  setDecisions: (decisions) => set({ acceptedDecisions: decisions }),

  setLastChangedColumns: (columns) => set({ lastChangedColumns: columns }),

  setQualityScores: (scores) => set({ qualityScores: scores }),

  reset: () => set(initialState),
    }),
    {
      // Survive a browser refresh within the tab; hydrated on the client (see StoreHydrator)
      name: "datamimic-workspace",
      storage: createJSONStorage(() => sessionStorage),
      skipHydration: true,
      partialize: (s) => ({
        fileId: s.fileId,
        columns: s.columns,
        columnInfo: s.columnInfo,
        rowCount: s.rowCount,
        generatedFileId: s.generatedFileId,
        generationHistory: s.generationHistory,
        currentVersionId: s.currentVersionId,
        kpiSummary: s.kpiSummary,
        demoAlgorithm: s.demoAlgorithm,
        exploreMode: s.exploreMode,
        planner: s.planner,
        selectedAlgorithm: s.selectedAlgorithm,
        generatorConfig: s.generatorConfig,
        pendingFeedback: s.pendingFeedback,
        acceptedDecisions: s.acceptedDecisions,
        lastChangedColumns: s.lastChangedColumns,
      }),
    }
  )
);

