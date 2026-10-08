import type {
  ActivityCliffResponse,
  CompoundPageResponse,
  ConflictRowResponse,
  DatasetProfileResponse,
  DatasetResponse,
  DescriptorProfileResponse,
  Direction,
  InvalidRowResponse,
  ScaffoldProfileResponse,
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
export type DatasetProfile = DatasetProfileResponse;
export type DescriptorProfile = DescriptorProfileResponse;
export type ScaffoldProfile = ScaffoldProfileResponse;
export type ActivityCliff = ActivityCliffResponse;
export type CompoundPage = CompoundPageResponse;
export type { Direction, SplitStrategy, TargetKind };

/**
 * Display names for the descriptors the profile computes. The backend emits
 * machine keys (`fraction_csp3`); a chemist reads "Fraction sp3". Keyed rather
 * than derived from the string, because "cLogP" and "TPSA" are not what any
 * title-casing of `clogp`/`tpsa` produces.
 */
export const DESCRIPTOR_LABELS: Record<string, string> = {
  molecular_weight: "Molecular weight",
  clogp: "cLogP",
  tpsa: "TPSA",
  hbd: "H-bond donors",
  hba: "H-bond acceptors",
  rotatable_bonds: "Rotatable bonds",
  aromatic_rings: "Aromatic rings",
  fraction_csp3: "Fraction sp3",
  heavy_atoms: "Heavy atoms",
};

export function descriptorLabel(name: string): string {
  return DESCRIPTOR_LABELS[name] ?? name;
}

/** One column the wizard will predict, with how it is measured. */
export interface DraftTarget {
  column: string;
  kind: TargetKind;
  unit: string;
  direction: Direction | "";
}

/** What the wizard accumulates. The file is held client-side until step 4. */
export interface DatasetDraft {
  file: File | null;
  name: string;
  structureColumn: string;
  /** In the order chosen; that order is kept on the dataset and everywhere after. */
  targets: DraftTarget[];
  strategy: SplitStrategy;
  seed: number;
  /** The column holding compound IDs, or null for none. */
  idColumn: string | null;
  /** For a predefined split: the column holding each row's partition. */
  splitColumn: string | null;
}

export const EMPTY_DRAFT: DatasetDraft = {
  file: null,
  name: "",
  structureColumn: "",
  targets: [],
  // Scaffold is the default on purpose. It is the pessimistic split, and the
  // whole product exists because a random split flatters a model that will
  // fail prospectively.
  strategy: "scaffold",
  seed: 42,
  idColumn: null,
  splitColumn: null,
};

/**
 * Plain-English consequences, shown at the point of choice. A split is a
 * scientific decision, not a setting, so the UI has to say what each one
 * actually simulates rather than just naming it.
 *
 * Key order is the order the wizard and the filter offer the strategies in,
 * because both enumerate this record rather than keeping a second list that
 * the next strategy could be left out of. Molecule splits first, sequence
 * splits after.
 */
export const SPLIT_COPY: Record<SplitStrategy, { title: string; detail: string }> = {
  scaffold: {
    title: "Scaffold",
    detail:
      "Test compounds have Bemis–Murcko scaffolds absent from training. This approximates prediction on a new chemical series and gives a conservative estimate.",
  },
  random: {
    title: "Random",
    detail:
      "Compounds or sequences are assigned at random, so close relatives of the training data appear in the test set. Scores typically overestimate prospective performance.",
  },
  identity: {
    title: "Identity",
    detail:
      "Test sequences belong to protein families absent from training. This approximates prediction on a protein the model has not seen and gives a conservative estimate.",
  },
  position: {
    title: "Position",
    detail:
      "Test variants are mutated at residue positions never mutated in training. This approximates prediction at a site in the protein that has not been tested and gives a conservative estimate.",
  },
  // Last on purpose: key order is presentation order, and this is the specialist
  // option. A scientist reaching for it already knows which column holds the answer.
  predefined: {
    title: "From a column in your file",
    detail:
      "Each row's partition is read from a column you choose, exactly as your file declares it. Use this to reproduce a published benchmark on its own training and test sets, so your result can be compared with theirs.",
  },
};

export const TARGET_KIND_COPY: Record<TargetKind, { title: string; detail: string }> = {
  numeric: {
    title: "A measured value",
    detail: "For example pIC50, log solubility or permeability.",
  },
  binary: {
    title: "Active or inactive",
    detail: "Values must be 0 (inactive) or 1 (active).",
  },
};
