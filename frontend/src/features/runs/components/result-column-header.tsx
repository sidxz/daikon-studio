import type { IHeaderParams } from "ag-grid-community";

export function resultTargetLabel(name: string) {
  const label = name.replace(/_/g, " ");
  // Preserve scientific names and acronyms, such as pIC50 and CYP3A4.
  return /^[a-z ]+$/.test(label) ? label.charAt(0).toUpperCase() + label.slice(1) : label;
}

type HeaderKind =
  | "structure"
  | "id"
  | "row"
  | "smiles"
  | "numeric"
  | "probability"
  | "class"
  | "uncertainty"
  | "applicability";

/** Replace only the label; AG Grid retains its sorting, filtering and keyboard controls. */
function ResultColumnHeader({
  displayName,
  title,
  subtitle,
}: IHeaderParams & { title?: string; subtitle: string }) {
  return (
    <div className="results-column-heading">
      <span className="results-column-copy">
        <span className="results-column-title">{title ?? displayName}</span>
        <span className="results-column-subtitle">{subtitle}</span>
      </span>
    </div>
  );
}

export function resultHeader(kind: HeaderKind, subtitle: string, title?: string) {
  return {
    headerClass: `results-header-${kind}`,
    headerComponentParams: {
      innerHeaderComponent: ResultColumnHeader,
      innerHeaderComponentParams: { title, subtitle },
    },
  };
}
