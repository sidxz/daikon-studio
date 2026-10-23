"use client";

import { StructureViewer } from "@/features/pages/components/renderers/structure-viewer";
import { StructureThumbnail } from "@/shared/components/chemistry/structure-thumbnail";
import { Button } from "@/shared/components/ui/button";
import { Skeleton } from "@/shared/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/shared/components/ui/table";
import { cn } from "@/shared/lib/utils";
import { type ReactNode, useMemo, useState } from "react";
import { useStudyValue } from "../hooks/use-studies";
import { RUN_UNDER_WAY, STEPS_REASON_ID, type ValueSummary } from "../types";
import { type PocketResidue, PocketViewer } from "./pocket-viewer";

type Row = Record<string, unknown>;

/** A page shows a look at a table, not the table; the rest is a download away. */
const SHOWN_ROWS = 20;
const STRUCTURE_COLUMN = /^(canonical_)?smiles$/i;

function cell(value: unknown): string {
  if (value == null) return "—";
  if (typeof value === "number")
    return value.toLocaleString(undefined, { maximumFractionDigits: 3 });
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function isRows(data: unknown): data is Row[] {
  return Array.isArray(data) && data.every((r) => typeof r === "object" && r !== null);
}

function asRecord(value: unknown): Row {
  return typeof value === "object" && value !== null ? (value as Row) : {};
}

function RowsTable({ rows, columns, total }: { rows: Row[]; columns: string[]; total?: number }) {
  const shown = rows.slice(0, SHOWN_ROWS);
  return (
    <div className="space-y-1">
      <div className="overflow-x-auto rounded-sm border">
        <Table>
          <TableHeader>
            <TableRow>
              {columns.map((column) => (
                <TableHead key={column}>{column}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {shown.map((row, index) => (
              // Rows have no identity of their own; their order is the table's.
              // biome-ignore lint/suspicious/noArrayIndexKey: a preview row is its position
              <TableRow key={index}>
                {columns.map((column) => (
                  <TableCell key={column}>
                    {STRUCTURE_COLUMN.test(column) && typeof row[column] === "string" ? (
                      <div className="flex items-center gap-2">
                        <StructureThumbnail smiles={row[column] as string} size={48} />
                        <span className="max-w-48 truncate font-mono text-xs">
                          {row[column] as string}
                        </span>
                      </div>
                    ) : (
                      cell(row[column])
                    )}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      {(total ?? rows.length) > shown.length && (
        <p className="text-xs text-muted-foreground">
          First {shown.length} of {(total ?? rows.length).toLocaleString()} rows.
        </p>
      )}
    </div>
  );
}

function Facts({ facts }: { facts: [string, unknown][] }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
      {facts
        .filter(([, value]) => value != null && value !== "")
        .map(([label, value]) => (
          <div key={label} className="contents">
            <dt className="text-muted-foreground">{label}</dt>
            <dd className="min-w-0 wrap-anywhere">{String(value)}</dd>
          </div>
        ))}
    </dl>
  );
}

function Structure({ summary }: { summary: Row }) {
  const ligands = Array.isArray(summary.ligands) ? (summary.ligands as Row[]) : [];
  // Once prepared, `ligands` are still the entry's own; what stayed in the receptor is
  // `kept_ligands`, and an empty list there means every one was stripped.
  const kept = Array.isArray(summary.kept_ligands) ? (summary.kept_ligands as string[]) : null;
  return (
    <div className="space-y-3">
      <Facts
        facts={[
          ["ID", summary.id],
          ["Title", summary.title],
          ["Method", summary.method],
          ["Resolution", summary.resolution != null ? `${cell(summary.resolution)} Å` : null],
          ["Chains", Array.isArray(summary.chains) ? summary.chains.join(", ") : null],
          [
            kept ? "Ligands before preparation" : "Ligands",
            ligands.length
              ? ligands
                  .map(
                    (l) =>
                      `${l.comp_id} (chain ${l.chain} ${l.resseq}, ${
                        Array.isArray(l.contacts) ? l.contacts.length : 0
                      } contacts)`,
                  )
                  .join("; ")
              : "None",
          ],
          ["Kept in the receptor", kept && (kept.join(", ") || "None")],
        ]}
      />
      {/* Mol* loads deposited entries by id; a predicted model has none to load. */}
      {summary.source === "rcsb" && typeof summary.id === "string" && (
        <StructureViewer source="pdb" sourceId={summary.id} width="half" />
      )}
    </div>
  );
}

function Correlation({ data }: { data: Row }) {
  const properties = isRows(data.properties) ? data.properties : [];
  const meta = asRecord(data.meta);
  const strongest = asRecord(meta.strongest).property;
  const excluded = typeof meta.excluded === "number" ? meta.excluded : 0;
  return (
    <div className="space-y-2">
      {excluded > 0 && (
        <p className="text-xs text-muted-foreground">
          {excluded} {excluded === 1 ? "reading" : "readings"} in {String(meta.activity_column)}{" "}
          could not be read as a number and {excluded === 1 ? "was" : "were"} left out.
        </p>
      )}
      <div className="overflow-x-auto rounded-sm border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Property</TableHead>
              <TableHead className="text-right">r</TableHead>
              <TableHead>95% interval</TableHead>
              <TableHead className="text-right">n</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {properties.map((p) => (
              <TableRow
                key={String(p.property)}
                className={cn(p.property === strongest && "font-semibold")}
              >
                <TableCell>{String(p.property)}</TableCell>
                <TableCell className="text-right">{cell(p.r)}</TableCell>
                <TableCell>
                  {p.ci_low != null && p.ci_high != null
                    ? `${cell(p.ci_low)} to ${cell(p.ci_high)}`
                    : "—"}
                </TableCell>
                <TableCell className="text-right">{cell(p.n)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

function Alerts({ data }: { data: Row[] }) {
  if (data.length === 0) {
    return <p className="text-sm text-muted-foreground">No compound was flagged.</p>;
  }
  return (
    <ul className="space-y-2">
      {data.map((flagged) => (
        <li key={String(flagged.row)} className="flex items-start gap-3 rounded-sm border p-2">
          {typeof flagged.smiles === "string" && (
            <StructureThumbnail smiles={flagged.smiles} size={48} />
          )}
          <div className="min-w-0 text-sm">
            <p className="font-medium">Row {cell(flagged.row)}</p>
            <ul className="text-muted-foreground">
              {(isRows(flagged.alerts) ? flagged.alerts : []).map((alert) => (
                <li key={`${alert.catalog}-${alert.description}`}>
                  {String(alert.description)}{" "}
                  <span className="text-xs">({String(alert.catalog)})</span>
                </li>
              ))}
            </ul>
          </div>
        </li>
      ))}
    </ul>
  );
}

function SiteAnnotations({ summary, data }: { summary: Row; data: Row }) {
  const features = isRows(data.features) ? data.features : [];
  return (
    <div className="space-y-3">
      <Facts
        facts={[
          ["Accession", summary.accession],
          ["Name", summary.name],
          ["Organism", summary.organism],
          ["Length", summary.length],
          ["Structures", summary.structures],
        ]}
      />
      {features.length > 0 && (
        <RowsTable
          columns={["type", "residues", "description"]}
          rows={features.map((f) => ({
            type: f.type,
            residues: f.start === f.end ? f.start : `${f.start}–${f.end}`,
            description: f.description,
          }))}
        />
      )}
    </div>
  );
}

/** A pocket's lining residues by chain and number, the identities a reader can check. */
function lining(pocket: Row): string {
  const residues = isRows(pocket.residues) ? pocket.residues : [];
  return residues.map((r) => `${r.chain}:${r.resname}${r.resseq}`).join(", ") || "—";
}

/** The selected pocket in words: the residue identities the table only counts,
 *  and the annotation descriptions behind its "Annotated" column. Without these
 *  a reader cannot check what the viewer is painting. */
function PocketDetail({ pocket, scored }: { pocket: Row; scored: string }) {
  const residues = isRows(pocket.residues) ? pocket.residues : [];
  const overlap = isRows(pocket.ligand_overlap) ? pocket.ligand_overlap : [];
  const annotated = isRows(pocket.annotated) ? pocket.annotated : [];
  const chains = [...new Set(residues.map((r) => String(r.chain)))];
  const lines = Array.isArray(pocket.lines_ligand) ? (pocket.lines_ligand as string[]) : [];
  return (
    <div className="min-w-0 flex-1 space-y-1 text-sm">
      <p>
        <span className="font-medium">Rank {cell(pocket.rank)}</span>
        {" · "}
        {scored} {cell(pocket[scored])}
        {" · "}
        {cell(pocket.volume)} Å³{" · "}
        {residues.length} residues on chain {chains.join(", ") || "—"}
        {lines.length > 0 && ` · against ${lines.join(", ")}`}
      </p>
      <p className="text-muted-foreground">
        <span className="text-foreground">Lining </span>
        {lining(pocket)}
      </p>
      {overlap.length > 0 && (
        <p className="text-muted-foreground">
          <span className="text-foreground">Ligand contacts </span>
          {overlap
            .map(
              (o) =>
                `${o.comp_id}${o.additive ? " (from the crystallisation drop)" : ""} ${o.chain} ${o.resseq}: ${o.shared} of ${o.of}`,
            )
            .join("; ")}
        </p>
      )}
      {annotated.length > 0 && (
        <p className="text-muted-foreground">
          <span className="text-foreground">Annotated </span>
          {annotated
            .map((a) =>
              [`${a.type} ${a.start}${a.end === a.start ? "" : `–${a.end}`}`, a.description]
                .filter(Boolean)
                .join(" — "),
            )
            .join("; ")}
        </p>
      )}
    </div>
  );
}

function residueKeys(pocket: Row): Set<string> {
  const residues = isRows(pocket.residues) ? pocket.residues : [];
  return new Set(residues.map((r) => `${r.chain}:${r.resseq}`));
}

function ligandKeys(pocket: Row): Set<string> {
  const overlap = isRows(pocket.ligand_overlap) ? pocket.ligand_overlap : [];
  return new Set(overlap.map((o) => `${o.comp_id}:${o.chain}:${o.resseq}`));
}

/** The candidates that touch this one, by a shared lining residue or a shared
 *  deposited ligand. fpocket splits an extended site — a dinucleotide cleft, a
 *  lipid channel — into several entries that a ranked table shows as unrelated
 *  rows pages apart. Drawing the neighbours is how you see that they are one
 *  site. Every other candidate at once just paints the whole protein. */
function neighboursOf(pockets: Row[], active: Row | undefined): Row[] {
  if (!active) return [];
  const residues = residueKeys(active);
  const ligands = ligandKeys(active);
  return pockets.filter(
    (p) =>
      p.rank !== active.rank &&
      ([...residueKeys(p)].some((k) => residues.has(k)) ||
        [...ligandKeys(p)].some((k) => ligands.has(k))),
  );
}

/**
 * A candidate the scientist chose: the value the rest of the study is expressed against. One
 * chosen on a result the step has since replaced stands for nothing now, since the same rank
 * may be another candidate, and is labelled as such rather than as the current choice.
 */
function Approved({ data, earlier }: { data: Row; earlier: boolean }) {
  const rank = cell(asRecord(data.meta).rank);
  return (
    <div className="space-y-1 text-sm">
      <p className={earlier ? "text-muted-foreground" : "font-medium"}>
        {earlier ? `Approved on an earlier result: rank ${rank}` : `Approved: rank ${rank}`}
      </p>
      <p className="text-muted-foreground">
        <span className="text-foreground">Lining </span>
        {lining(data)}
      </p>
    </div>
  );
}

function Pockets({
  data,
  onApprove,
  approveBlocked = null,
}: {
  data: Row;
  onApprove?: (rank: number) => void;
  approveBlocked?: string | null;
}) {
  const meta = asRecord(data.meta);
  const pockets = isRows(data.pockets) ? data.pockets : [];
  const ranks = Array.isArray(meta.ligand_site_ranks) ? (meta.ligand_site_ranks as number[]) : [];
  const structureId = typeof meta.structure_id === "string" ? meta.structure_id : null;
  const receptor = Array.isArray(meta.receptor_ligands) ? (meta.receptor_ligands as string[]) : [];
  const ignored = Array.isArray(meta.ignored_ligands) ? (meta.ignored_ligands as string[]) : [];
  // A predicted site carries a probability; a cavity, a druggability score. Different numbers.
  const scored = pockets.some((p) => "probability" in p) ? "probability" : "druggability";
  const numbering = asRecord(meta.numbering);
  // Said plainly, because every entry in the Annotated column depends on it and
  // an empty column otherwise looks like "nothing annotated here".
  const numberingNote = !numbering.checked
    ? "The file's residue numbering was not checked against UniProt, so no annotated sites are labelled."
    : numbering.matches
      ? `Residue numbering agrees with UniProt at ${cell(numbering.agree)} of ${cell(numbering.of)} residues.`
      : `Residue numbering disagrees with UniProt (${cell(numbering.agree)} of ${cell(numbering.of)} residues match)${
          numbering.offset == null
            ? ", and no constant offset reconciles them"
            : `; an offset of ${cell(numbering.offset)} would reconcile them`
        }. Annotated sites are not labelled.`;
  const count = pockets.length;
  const [selected, setSelected] = useState<number | null>(null);
  const [showOthers, setShowOthers] = useState(false);
  // The table's own top row is the neutral default; which pocket is worth
  // aiming at is the question the step gates on, not one to answer here.
  const active = pockets.find((p) => p.rank === selected) ?? pockets[0];
  // Memoised on the pocket: a fresh array every render would re-aim the camera
  // mid-drag.
  const activeResidues: PocketResidue[] = useMemo(
    () =>
      (active && isRows(active.residues) ? active.residues : []).map((r) => ({
        chain: String(r.chain),
        resname: String(r.resname),
        resseq: Number(r.resseq),
      })),
    [active],
  );
  const neighbours = useMemo(() => neighboursOf(pockets, active), [pockets, active]);
  const otherResidues: PocketResidue[] = useMemo(
    () =>
      showOthers
        ? neighbours
            .flatMap((p) => (isRows(p.residues) ? p.residues : []))
            .map((r) => ({
              chain: String(r.chain),
              resname: String(r.resname),
              resseq: Number(r.resseq),
            }))
        : [],
    [showOthers, neighbours],
  );
  return (
    <div className="space-y-2">
      <p className="text-sm">
        {count} {count === 1 ? "pocket" : "pockets"}.
        {ranks.length > 0 &&
          ` ${ranks.length === 1 ? "Rank" : "Ranks"} ${ranks.join(", ")} ${
            ranks.length === 1 ? "overlaps" : "overlap"
          } a deposited ligand's contacts.`}{" "}
        {receptor.length > 0
          ? `${receptor.join(", ")} stayed in place as part of the receptor, so these are the cavities left beside ${receptor.length === 1 ? "it" : "them"}.`
          : "Nothing was kept as a cofactor, so a cavity a cofactor would fill is reported as a pocket."}
        {ignored.length > 0 && ` Ignored by the predictor: ${ignored.join(", ")}.`}
        {scored === "druggability" &&
          " Volumes are Monte Carlo estimates and move a few percent between runs."}
      </p>
      <p className="text-sm text-muted-foreground">{numberingNote}</p>
      {structureId && active && (
        <div className="space-y-2">
          <PocketViewer pdbId={structureId} residues={activeResidues} others={otherResidues} />
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={showOthers}
              disabled={neighbours.length === 0}
              onChange={(e) => setShowOthers(e.target.checked)}
            />
            {neighbours.length > 0
              ? `Show the ${neighbours.length} candidate${neighbours.length === 1 ? "" : "s"} touching this one, in amber (rank${neighbours.length === 1 ? "" : "s"} ${neighbours.map((p) => cell(p.rank)).join(", ")})`
              : "No other candidate touches this one"}
          </label>
          <p className="text-xs text-muted-foreground">
            Deposited entry {structureId.toUpperCase()} with its ligands, and the lining residues of
            rank {cell(active.rank)} in magenta. fpocket ran on the prepared structure, which has no
            ligands or waters. Pick a row to aim the view at another pocket.
          </p>
        </div>
      )}
      {/* Outside the viewer: a structure uploaded as a file has no entry for it to load, and
          its pockets still need reading and choosing. */}
      {active && (
        <div className="flex flex-wrap items-start gap-3">
          <PocketDetail pocket={active} scored={scored} />
          {onApprove && (
            <div className="space-y-1">
              <Button
                size="sm"
                variant="outline"
                disabled={Boolean(approveBlocked)}
                // A run is said once, under the steps heading; any other reason, here.
                aria-describedby={approveBlocked === RUN_UNDER_WAY ? STEPS_REASON_ID : undefined}
                onClick={() => onApprove(Number(active.rank))}
              >
                Use this one
              </Button>
              {approveBlocked && approveBlocked !== RUN_UNDER_WAY && (
                <p className="max-w-48 text-xs text-muted-foreground">{approveBlocked}</p>
              )}
            </div>
          )}
        </div>
      )}
      <div className="max-h-[26rem] overflow-auto rounded-sm border">
        <Table>
          <TableHeader className="sticky top-0 z-10 bg-background">
            <TableRow>
              <TableHead>Rank</TableHead>
              <TableHead className="text-right">
                {scored === "probability" ? "Probability" : "Druggability"}
              </TableHead>
              <TableHead className="text-right">Score</TableHead>
              <TableHead className="text-right">Volume (Å³)</TableHead>
              <TableHead className="text-right">Residues</TableHead>
              <TableHead>Ligand overlap</TableHead>
              <TableHead>Annotated</TableHead>
              <TableHead>Interface</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {pockets.map((p) => {
              const overlap = isRows(p.ligand_overlap) ? p.ligand_overlap : [];
              const residues = isRows(p.residues) ? p.residues : [];
              const annotated = isRows(p.annotated) ? p.annotated : [];
              const isActive = active?.rank === p.rank;
              return (
                <TableRow
                  key={String(p.rank)}
                  onClick={() => setSelected(Number(p.rank))}
                  className={cn(
                    "cursor-pointer",
                    overlap.length > 0 && "font-bold",
                    isActive && "bg-muted",
                  )}
                >
                  <TableCell>
                    {/* The focusable control, so the view is reachable by keyboard
                        as well as by clicking anywhere on the row. */}
                    <button
                      type="button"
                      aria-pressed={isActive}
                      aria-label={`Show pocket rank ${cell(p.rank)} in the viewer`}
                      className="underline-offset-4 hover:underline focus-visible:underline"
                      onClick={() => setSelected(Number(p.rank))}
                    >
                      {cell(p.rank)}
                    </button>
                  </TableCell>
                  <TableCell className="text-right">{cell(p[scored])}</TableCell>
                  <TableCell className="text-right">{cell(p.score)}</TableCell>
                  <TableCell className="text-right">{cell(p.volume)}</TableCell>
                  <TableCell
                    className="text-right"
                    title={residues
                      .slice(0, 6)
                      .map((r) => `${r.chain}:${r.resname} ${r.resseq}`)
                      .join(", ")}
                  >
                    {residues.length}
                  </TableCell>
                  <TableCell>
                    {overlap.length
                      ? overlap
                          .map(
                            (o) =>
                              `${o.comp_id}${o.additive ? " (additive)" : ""} ${o.shared}/${o.of}`,
                          )
                          .join(", ")
                      : "—"}
                  </TableCell>
                  <TableCell>
                    {annotated.length ? [...new Set(annotated.map((a) => a.type))].join(", ") : "—"}
                  </TableCell>
                  <TableCell>{p.interface ? "Yes" : "No"}</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

/**
 * One value a step produced or the scientist brought, read only when shown. `onApprove` offers
 * the choice of a reported candidate, which only a finished judgment step can take, and
 * `approveBlocked` says why it cannot be taken right now. `earlier` marks a choice made on a
 * result its step has since replaced.
 */
export function ValueBlock({
  studyId,
  value,
  onApprove,
  approveBlocked = null,
  earlier = false,
}: {
  studyId: string;
  value: ValueSummary;
  onApprove?: (rank: number) => void;
  approveBlocked?: string | null;
  earlier?: boolean;
}) {
  const query = useStudyValue(studyId, value.id);
  const read = query.data;

  let body: ReactNode;
  if (query.isError) {
    body = <p className="text-sm text-destructive">Could not load this result.</p>;
  } else if (!read) {
    body = <Skeleton className="h-24 w-full" />;
  } else if (read.kind === "structure") {
    body = <Structure summary={read.summary} />;
  } else if (read.kind === "correlation") {
    body = <Correlation data={asRecord(read.data)} />;
  } else if (read.kind === "alerts" && isRows(read.data)) {
    body = <Alerts data={read.data} />;
  } else if (read.kind === "site_annotations") {
    body = <SiteAnnotations summary={read.summary} data={asRecord(read.data)} />;
  } else if (read.kind === "pocket_candidates") {
    body = (
      <Pockets data={asRecord(read.data)} onApprove={onApprove} approveBlocked={approveBlocked} />
    );
  } else if (read.kind === "pocket") {
    body = <Approved data={asRecord(read.data)} earlier={earlier} />;
  } else if (isRows(read.data) && read.data.length > 0) {
    const columns = Array.isArray(read.summary.columns)
      ? (read.summary.columns as string[])
      : Object.keys(read.data[0]);
    const total = typeof read.summary.rows === "number" ? read.summary.rows : undefined;
    body = <RowsTable rows={read.data} columns={columns} total={total} />;
  } else {
    body = (
      <pre className="max-h-80 overflow-auto rounded-sm bg-muted p-3 text-xs">
        {JSON.stringify(read.data, null, 2)}
      </pre>
    );
  }

  return (
    <section className="min-w-0 space-y-2">
      <h4 className="text-xs uppercase tracking-wide text-muted-foreground">
        {value.key.replace(/_/g, " ")}
      </h4>
      {body}
    </section>
  );
}
