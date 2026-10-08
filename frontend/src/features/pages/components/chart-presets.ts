import type { ChartKind, ChartOptions } from "./renderers/chart-figure";

/**
 * The domain charts of early drug discovery and ML-result interpretation. Each
 * is a plain configuration over the five marks the renderer supports, not its
 * own code path — a volcano plot is a scatter with threshold lines, an ROC
 * curve is a line with a chance diagonal.
 *
 * `sample` is shown as the data textarea's placeholder so the expected column
 * shape is visible before the author types anything. Every sample must parse as
 * comma-delimited: the delimiter sniff in parseTable is table-wide, so one stray
 * tab would make the whole sample read as a single column.
 */
export type ChartPreset = {
  label: string;
  hint: string;
  keywords: string[];
  kind: ChartKind;
  options: ChartOptions;
  sample: string;
};

export const CHART_PRESETS: Record<string, ChartPreset> = {
  volcano: {
    label: "Volcano plot",
    hint: "Fold change against significance",
    // Deliberately no "proteomics": it contains "prot", which would make the
    // Protein embed no longer the only hit for that query.
    keywords: ["volcano", "differential", "expression", "rnaseq"],
    kind: "scatter",
    options: {
      refLines: [
        { axis: "x", value: -1 },
        { axis: "x", value: 1 },
        { axis: "y", value: 1.3, label: "p = 0.05" },
      ],
    },
    sample: "log2fc,neglog10p\n-2.4,4.1\n-0.3,0.2\n1.8,3.6",
  },
  roc: {
    label: "ROC curve",
    hint: "True positive rate against false positive rate",
    keywords: ["roc", "auc", "classifier", "sensitivity", "specificity", "ml"],
    kind: "line",
    options: { refLines: [{ segment: "diagonal" }] },
    sample: "fpr,tpr\n0,0\n0.1,0.62\n0.35,0.88\n1,1",
  },
  doseResponse: {
    label: "Dose-response",
    hint: "Response against a log concentration axis",
    keywords: ["dose", "response", "ic50", "ec50", "potency", "curve"],
    // Concentration spans orders of magnitude, so the x axis is log by
    // definition. No curve is fitted — plot measured points, or paste a fitted
    // column alongside them. The sample carries an `sd` column so the error-bar
    // picker has something to offer, but the preset doesn't preselect it: a
    // stored column name the author's own data lacks would leave the picker
    // showing a value that isn't in its list.
    options: { logX: true },
    kind: "scatter",
    sample: "conc_nm,response,sd\n1,4,1\n10,22,3\n100,71,5\n1000,96,2",
  },
  parity: {
    label: "Parity plot",
    hint: "Predicted against observed, with the y=x line",
    keywords: ["parity", "predicted", "observed", "calibration", "regression", "ml"],
    kind: "scatter",
    options: { refLines: [{ segment: "identity" }] },
    sample: "observed,predicted\n5.1,5.4\n6.8,6.2\n7.9,8.1",
  },
  featureImportance: {
    label: "Feature importance",
    hint: "Horizontal bars, one per feature",
    keywords: ["feature", "importance", "shap", "model", "ml"],
    kind: "bar",
    options: { horizontal: true },
    sample: "feature,importance\nlogP,0.31\nTPSA,0.24\nHBD,0.11",
  },
  kaplanMeier: {
    label: "Kaplan-Meier",
    hint: "Step survival curve over time",
    keywords: ["kaplan", "meier", "survival", "time", "event"],
    kind: "line",
    options: { step: true },
    sample: "months,survival\n0,1\n6,0.92\n12,0.74\n24,0.51",
  },
  propertyProfile: {
    label: "Property profile",
    hint: "Radar across several properties",
    keywords: ["radar", "profile", "property", "physchem", "spider", "lead"],
    kind: "radar",
    options: {},
    sample: "property,compound A,compound B\nlogP,3.1,4.2\nTPSA,72,58\nMW,342,410",
  },
  composition: {
    label: "Composition",
    hint: "Part-to-whole breakdown",
    keywords: ["pie", "composition", "breakdown", "share", "proportion"],
    kind: "pie",
    options: {},
    sample: "class,count\nkinase,42\nprotease,18\nGPCR,9",
  },
};
