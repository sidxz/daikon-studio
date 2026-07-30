import type {
  ConflictRowResponse,
  DatasetResponse,
  Direction,
  InvalidRowResponse,
  SplitBody,
  SplitStrategy,
  TargetBody,
  TargetKind,
  ValidationReportResponse,
} from "@/shared/lib/api/model";

// Generated DTOs under domain names. Never redeclare their shape.
export type Dataset = DatasetResponse;
export type ValidationReport = ValidationReportResponse;
export type InvalidRow = InvalidRowResponse;
export type ConflictRow = ConflictRowResponse;
export type TargetSpec = TargetBody;
export type SplitSpec = SplitBody;
export type { Direction, SplitStrategy, TargetKind };

/** What the wizard accumulates. The file is held client-side until step 4. */
export interface DatasetDraft {
  file: File | null;
  name: string;
  structureColumn: string;
  targetColumn: string;
  kind: TargetKind;
  unit: string;
  direction: Direction | "";
  strategy: SplitStrategy;
  seed: number;
}

export const EMPTY_DRAFT: DatasetDraft = {
  file: null,
  name: "",
  structureColumn: "",
  targetColumn: "",
  kind: "numeric",
  unit: "",
  direction: "high",
  // Scaffold is the default on purpose. It is the pessimistic split, and the
  // whole product exists because a random split flatters a model that will
  // fail prospectively.
  strategy: "scaffold",
  seed: 42,
};

/**
 * Plain-English consequences, shown at the point of choice. A split is a
 * scientific decision, not a setting, so the UI has to say what each one
 * actually simulates rather than just naming it.
 */
export const SPLIT_COPY: Record<SplitStrategy, { title: string; detail: string }> = {
  scaffold: {
    title: "Scaffold",
    detail:
      "Test compounds have chemical scaffolds the model never trained on. This simulates asking the model about a new series — the situation you are usually in, and the harder score to earn.",
  },
  random: {
    title: "Random",
    detail:
      "Test compounds are drawn at random, so close analogues of training compounds end up on both sides. Scores come out higher than the model will achieve prospectively.",
  },
};

export const TARGET_KIND_COPY: Record<TargetKind, { title: string; detail: string }> = {
  numeric: {
    title: "A measured value",
    detail: "IC50, solubility, permeability — anything on a continuous scale.",
  },
  binary: {
    title: "Active or inactive",
    detail: "A two-class call. The column must hold exactly two distinct values.",
  },
};
