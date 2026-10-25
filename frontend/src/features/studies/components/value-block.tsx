"use client";

import { StructureViewer } from "@/features/pages/components/renderers/structure-viewer";
import { ForestChart } from "@/shared/components/charts";
import { StructureThumbnail } from "@/shared/components/chemistry/structure-thumbnail";
import { Badge } from "@/shared/components/ui/badge";
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
import { useStudyValue, useStudyValueFile } from "../hooks/use-studies";
import { RUN_UNDER_WAY, STEPS_REASON_ID, type ValueSummary } from "../types";
import { type PocketResidue, PocketViewer } from "./pocket-viewer";
import { PoseViewer } from "./pose-viewer";

type Row = Record<string, unknown>;

/** A page shows a look at a table, not the table; the rest is a download away. */
const SHOWN_ROWS = 20;
// Columns that hold a drawable structure. `scaffold` and `scaffold_generic` earn their
// place here for the reason the backend withholds them from the agent: they are
// structures, and the one thing that defines a chemotype should not be the one thing
// shown as a string of brackets.
const STRUCTURE_COLUMN = /^(canonical_)?smiles$|^scaffold(_generic)?$/i;

/**
 * Past this, a cell is a payload rather than a value, and its size says more than any
 * prefix of it would. A docked pose carries kilobytes of coordinates per row: shown as
 * text it buries every column beside it, and the first eighty characters of it tell a
 * reader nothing. The value still holds it; there is nowhere yet to send a reader for it,
 * which is the gap a pose viewer and a file download both close.
 */
const CELL_MAX = 200;

function cell(value: unknown): string {
  if (value == null) return "—";
  if (typeof value === "number")
    return value.toLocaleString(undefined, { maximumFractionDigits: 3 });
  const text = typeof value === "object" ? JSON.stringify(value) : String(value);
  if (text.length <= CELL_MAX) return text;
  return `${(text.length / 1024).toFixed(1)} kB, not shown`;
}

function isRows(data: unknown): data is Row[] {
  return Array.isArray(data) && data.every((r) => typeof r === "object" && r !== null);
}

function asRecord(value: unknown): Row {
  return typeof value === "object" && value !== null ? (value as Row) : {};
}

function RowsTable({
  rows,
  columns,
  total,
  override,
  highlight,
}: {
  rows: Row[];
  columns: string[];
  total?: number;
  /** A SMARTS coloured on every drawn structure, so a match can be seen and not just asserted. */
  highlight?: string;
  /** A cell drawn in place of its text; undefined leaves the text. */
  override?: (row: Row, column: string) => ReactNode;
}) {
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
                    {override?.(row, column) ??
                      (STRUCTURE_COLUMN.test(column) && typeof row[column] === "string" ? (
                        <div className="flex items-center gap-2">
                          <StructureThumbnail
                            smiles={row[column] as string}
                            size={48}
                            highlight={highlight}
                          />
                          <span className="max-w-48 truncate font-mono text-xs">
                            {row[column] as string}
                          </span>
                        </div>
                      ) : (
                        cell(row[column])
                      ))}
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
          // Named, not counted: whether a gap matters is decided by where it sits.
          [
            "Residues unresolved in the experimental structure",
            Array.isArray(summary.missing_residues)
              ? (summary.missing_residues as string[]).join(", ") || "None"
              : null,
          ],
          // A preparation gate is there to be read before anything docks into the result.
          // The pKa list behind this is fifty-odd records; what a reader needs is the
          // residues that came out against expectation, and the sentence that says what
          // the silence about the others means.
          [
            "Protonation differing from the usual state",
            Array.isArray(summary.protonation_shifted)
              ? (summary.protonation_shifted as string[]).join("; ") || "None"
              : null,
          ],
          ["Protonation", summary.protonation],
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
          {excluded} nonnumeric {excluded === 1 ? "value" : "values"} in{" "}
          {String(meta.activity_column)} {excluded === 1 ? "was" : "were"} excluded from the
          analysis.
        </p>
      )}
      {properties.length > 0 && (
        <ForestChart
          data={properties.map((p) => ({
            label: String(p.property),
            value: typeof p.r === "number" ? p.r : 0,
            low: typeof p.ci_low === "number" ? p.ci_low : null,
            high: typeof p.ci_high === "number" ? p.ci_high : null,
          }))}
          xLabel="Correlation with activity (r)"
          height={Math.max(160, properties.length * 26 + 60)}
          caption="An interval crossing zero is a property activity does not track."
        />
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
                .join(": "),
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
function Approved({
  data,
  earlier,
  children,
}: {
  data: Row;
  earlier: boolean;
  children?: ReactNode;
}) {
  const rank = cell(asRecord(data.meta).rank);
  return (
    <div className="space-y-1 text-sm">
      <p className={earlier ? "text-muted-foreground" : "font-medium"}>
        {earlier ? `Approved on an earlier result: rank ${rank}` : `Approved: rank ${rank}`}
      </p>
      {children ?? (
        <p className="text-muted-foreground">
          <span className="text-foreground">Lining </span>
          {lining(data)}
        </p>
      )}
    </div>
  );
}

/** A server-made SVG can still carry user text (a core typed by the scientist, an uploaded
 *  column), so it is shown as an image: scripts never run in an SVG loaded through `<img>`. */
function Drawing({
  svg,
  className,
  label = "Core drawing",
}: {
  svg: unknown;
  className?: string;
  label?: string;
}) {
  if (typeof svg !== "string") return null;
  return (
    <img
      src={`data:image/svg+xml;utf8,${encodeURIComponent(svg)}`}
      alt={label}
      className={cn("h-auto w-full", className)}
    />
  );
}

function variableText(core: Row): string {
  return Object.entries(asRecord(core.variable))
    .map(([position, groups]) => `${position}: ${String(groups)}`)
    .join("; ");
}

function ApprovedCore({ data, earlier }: { data: Row; earlier: boolean }) {
  const description = Array.isArray(data.description) ? (data.description as string[]) : [];
  return (
    <Approved data={data} earlier={earlier}>
      <div className="space-y-1">
        <Drawing svg={data.svg} className="w-56" />
        {description.map((line) => (
          <p key={line} className="text-muted-foreground">
            {line}
          </p>
        ))}
      </div>
    </Approved>
  );
}

function Cores({
  data,
  onApprove,
  approveBlocked = null,
}: {
  data: Row;
  onApprove?: (rank: number) => void;
  approveBlocked?: string | null;
}) {
  const cores = isRows(data.candidates) ? data.candidates : [];
  const [selected, setSelected] = useState<number | null>(null);
  const active = cores.find((c) => c.rank === selected) ?? cores[0];
  const hasSeries = cores.some((c) => c.series != null);
  const description =
    active && Array.isArray(active.description) ? (active.description as string[]) : [];
  const misses = active && isRows(active.misses) ? active.misses : [];
  return (
    <div className="space-y-2">
      <p className="text-sm">
        {cores.length} candidate {cores.length === 1 ? "core" : "cores"}.
      </p>
      {active && (
        <div className="flex flex-wrap items-start gap-3">
          <Drawing svg={active.svg} className="w-56" />
          <div className="min-w-0 flex-1 space-y-1 text-sm">
            <p className="font-medium">Rank {cell(active.rank)}</p>
            {typeof active.preset_label === "string" && (
              <p className="text-muted-foreground">{active.preset_label}</p>
            )}
            {description.map((line) => (
              <p key={line} className="text-muted-foreground">
                {line}
              </p>
            ))}
            {active.timed_out === true && (
              <p className="text-muted-foreground">
                The core search reached its time limit; a larger common core may exist.
              </p>
            )}
            {misses.length > 0 && (
              <div>
                <p className="text-foreground">Unmatched compounds</p>
                <ul className="text-muted-foreground">
                  {misses.map((m) => (
                    <li key={String(m.id)}>
                      {String(m.id)}: {String(m.reason)}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
          {onApprove && (
            <div className="space-y-1">
              <Button
                size="sm"
                variant="outline"
                disabled={Boolean(approveBlocked)}
                aria-describedby={approveBlocked === RUN_UNDER_WAY ? STEPS_REASON_ID : undefined}
                onClick={() => onApprove(Number(active.rank))}
              >
                Select this core
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
              <TableHead>Core</TableHead>
              {hasSeries && <TableHead>Series</TableHead>}
              <TableHead className="text-right">Coverage</TableHead>
              <TableHead className="text-right">Atoms</TableHead>
              <TableHead>Variable</TableHead>
              <TableHead className="text-right">Unmatched</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {cores.map((c) => {
              const isActive = active?.rank === c.rank;
              const unmatched = Array.isArray(c.unmatched) ? (c.unmatched as unknown[]) : [];
              return (
                <TableRow
                  key={String(c.rank)}
                  onClick={() => setSelected(Number(c.rank))}
                  className={cn("cursor-pointer", isActive && "bg-muted")}
                >
                  <TableCell>
                    <button
                      type="button"
                      aria-pressed={isActive}
                      aria-label={`Show core rank ${cell(c.rank)}`}
                      className="underline-offset-4 hover:underline focus-visible:underline"
                      onClick={() => setSelected(Number(c.rank))}
                    >
                      {cell(c.rank)}
                    </button>
                  </TableCell>
                  <TableCell>
                    <Drawing svg={c.svg} className="w-24" />
                  </TableCell>
                  {hasSeries && <TableCell>{cell(c.series)}</TableCell>}
                  <TableCell className="text-right">
                    {typeof c.coverage === "number" ? `${Math.round(c.coverage * 100)}%` : "—"}
                  </TableCell>
                  <TableCell className="text-right">{cell(c.n_atoms)}</TableCell>
                  <TableCell>{variableText(c) || "—"}</TableCell>
                  <TableCell className="text-right" title={unmatched.map(String).join(", ")}>
                    {unmatched.length}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

const R_COLUMN = /^R\d+$/;
const R_HELPER_COLUMN = /^R\d+_(svg|smiles)$/;

/** The SAR table: a substituent with no conventional name arrives drawn, in `R{n}_svg`. */
function RGroupTable({ rows, columns, total }: { rows: Row[]; columns: string[]; total?: number }) {
  return (
    <RowsTable
      rows={rows}
      columns={columns.filter((c) => !R_HELPER_COLUMN.test(c))}
      total={total}
      override={(row, column) =>
        R_COLUMN.test(column) && typeof row[`${column}_svg`] === "string" ? (
          <Drawing svg={row[`${column}_svg`]} className="w-16" label={`${column} drawing`} />
        ) : undefined
      }
    />
  );
}

/**
 * What each change is worth, with the interval around it.
 *
 * A transform table sorted by mean delta reads as a ranking of ideas, which is exactly
 * the mistake: a change measured on three pairs and a change measured on forty look the
 * same in a column. Drawn, the wide interval says "we do not know this yet" without
 * anyone having to check n, and a bar crossing zero stops the claim.
 */
/**
 * Compounds grouped by what they share, drawn.
 *
 * A clustering or a scaffold assignment answers "which of these belong together", and
 * the answer is a column of integers. No chemist can agree or disagree with an integer.
 * Shown as the structures themselves, grouped, with the representative named, the call
 * becomes reviewable in the way the gate on it assumes: would I have put these together.
 */
function GroupedStructures({
  rows,
  groupColumn,
  structureColumn,
  centroidColumn,
  label,
}: {
  rows: Row[];
  groupColumn: string;
  structureColumn: string;
  centroidColumn?: string;
  label: string;
}) {
  const groups = useMemo(() => {
    const by = new Map<string, Row[]>();
    for (const row of rows) {
      const key = row[groupColumn];
      if (key == null) continue;
      const id = String(key);
      by.set(id, [...(by.get(id) ?? []), row]);
    }
    // Biggest first: a singleton is not a chemotype and should not head the list.
    return [...by.entries()].sort((a, b) => b[1].length - a[1].length).slice(0, 12);
  }, [rows, groupColumn]);
  const ungrouped = rows.filter((r) => r[groupColumn] == null).length;
  if (groups.length === 0) return null;
  return (
    <div className="space-y-3">
      {groups.map(([id, members]) => (
        <div key={id} className="rounded-sm border p-2">
          <p className="mb-1 text-xs text-muted-foreground">
            {label} {id} · {members.length} {members.length === 1 ? "compound" : "compounds"}
          </p>
          <div className="flex flex-wrap gap-2">
            {members.slice(0, 16).map((row, i) => {
              const centroid = centroidColumn ? row[centroidColumn] === true : false;
              return (
                // biome-ignore lint/suspicious/noArrayIndexKey: a preview tile is its position
                <div key={i} className="flex flex-col items-center">
                  <div className={cn("rounded-sm", centroid && "ring-2 ring-primary")}>
                    <StructureThumbnail smiles={String(row[structureColumn] ?? "")} size={64} />
                  </div>
                  <span className="max-w-20 truncate text-[10px] text-muted-foreground">
                    {String(row.compound_id ?? row.name ?? row.id ?? "")}
                    {centroid ? " ·" : ""}
                  </span>
                </div>
              );
            })}
            {members.length > 16 && (
              <span className="self-center text-xs text-muted-foreground">
                +{members.length - 16} more
              </span>
            )}
          </div>
        </div>
      ))}
      {ungrouped > 0 && (
        <p className="text-xs text-muted-foreground">
          {ungrouped} {ungrouped === 1 ? "compound has" : "compounds have"} no {label.toLowerCase()}
          .
        </p>
      )}
    </div>
  );
}

/**
 * Docked poses: the ranking, and the pose a reader picked, drawn in the receptor.
 *
 * The table answers which compound scored best. Only the picture answers whether the
 * pose is worth believing, which is the question that matters most here, because a Vina
 * affinity is the least reliable number in the whole product.
 */
function Poses({
  rows,
  columns,
  total,
  receptor,
}: {
  rows: Row[];
  columns: string[];
  total?: number;
  receptor?: string;
}) {
  const [picked, setPicked] = useState(0);
  const pose = rows[picked];
  const geometry = typeof pose?.pose_pdbqt === "string" ? pose.pose_pdbqt : undefined;
  return (
    <div className="space-y-2">
      {receptor && geometry ? (
        <div className="space-y-1">
          <PoseViewer
            receptor={receptor}
            pose={geometry}
            name={`${String(pose?.compound ?? "pose")}-${String(pose?.pose_rank ?? "")}`}
          />
          <div className="flex flex-wrap items-center gap-1">
            {rows.slice(0, 12).map((row, i) => (
              <Button
                // biome-ignore lint/suspicious/noArrayIndexKey: a pose is its row
                key={i}
                size="sm"
                variant={i === picked ? "secondary" : "ghost"}
                onClick={() => setPicked(i)}
              >
                {String(row.compound ?? i + 1)} #{String(row.pose_rank ?? i + 1)}
              </Button>
            ))}
          </div>
        </div>
      ) : (
        <p className="text-xs text-muted-foreground">
          {geometry
            ? "The receptor this was docked into is not on this step, so the pose cannot be drawn."
            : "No pose geometry was returned."}
        </p>
      )}
      <RowsTable rows={rows} columns={columns} total={total} />
    </div>
  );
}

function Effects({ rows, columns, total }: { rows: Row[]; columns: string[]; total?: number }) {
  const shown = rows
    .filter((r) => typeof r.mean_delta === "number")
    .sort((a, b) => Math.abs(Number(b.mean_delta)) - Math.abs(Number(a.mean_delta)))
    .slice(0, 20);
  return (
    <div className="space-y-2">
      {shown.length > 0 && (
        <ForestChart
          data={shown.map((r) => ({
            label: `${String(r.position)} ${String(r.from)}\u2192${String(r.to)}`,
            value: Number(r.mean_delta),
            low: typeof r.ci_low === "number" ? r.ci_low : null,
            high: typeof r.ci_high === "number" ? r.ci_high : null,
          }))}
          xLabel="Mean change in activity"
          height={Math.max(180, shown.length * 24 + 60)}
          caption="The strongest changes by size of effect. An interval crossing zero is a change this series cannot yet distinguish from none."
        />
      )}
      <RowsTable rows={rows} columns={columns} total={total} />
    </div>
  );
}

/**
 * The kinds ProLIF names, in the order a reader weighs them: a directional bond first,
 * then a charge pairing or a ring stacking, then mere proximity. A kind not named here
 * sits with the hydrophobic ones, because a catalogue of contact types is not something
 * this view should be the authority on.
 */
const CONTACT_WEIGHT: Record<string, number> = {
  HBDonor: 0,
  HBAcceptor: 0,
  Anionic: 1,
  Cationic: 1,
  PiStacking: 1,
  FaceToFace: 1,
  EdgeToFace: 1,
  CationPi: 1,
  PiCation: 1,
  XBDonor: 1,
  XBAcceptor: 1,
  Hydrophobic: 2,
  VdWContact: 3,
};

function contactRank(kind: string): number {
  return CONTACT_WEIGHT[kind] ?? 2;
}

/**
 * The contacts a pose makes, read as residues rather than as one row per contact.
 *
 * The table underneath says the same thing in a form nobody can hold against the
 * literature: a reader asking whether indinavir reaches Asp25 should not have to count
 * rows to find out. Residues are ordered by the strongest contact each one makes, so the
 * hydrogen bonds come first, which is where the question usually is.
 */
function Interactions({
  rows,
  columns,
  total,
}: {
  rows: Row[];
  columns: string[];
  total?: number;
}) {
  const poses = useMemo(() => {
    const byPose = new Map<string, Map<string, Set<string>>>();
    for (const row of rows) {
      const pose = `${String(row.compound ?? "")} \u00b7 pose ${String(row.pose_rank ?? "")}`;
      const residue = row.chain
        ? `${String(row.residue)} ${String(row.chain)}`
        : String(row.residue);
      const residues = byPose.get(pose) ?? new Map<string, Set<string>>();
      byPose.set(pose, residues);
      const kinds = residues.get(residue) ?? new Set<string>();
      residues.set(residue, kinds);
      kinds.add(String(row.interaction));
    }
    return [...byPose].map(([pose, residues]) => ({
      pose,
      residues: [...residues]
        .map(([residue, kinds]) => ({
          residue,
          kinds: [...kinds].sort((a, b) => contactRank(a) - contactRank(b)),
        }))
        .sort(
          (a, b) =>
            contactRank(a.kinds[0]) - contactRank(b.kinds[0]) || a.residue.localeCompare(b.residue),
        ),
    }));
  }, [rows]);
  return (
    <div className="space-y-3">
      {poses.map(({ pose, residues }) => (
        <div key={pose} className="space-y-1">
          <p className="font-medium text-sm">
            {pose} &middot; {residues.length} {residues.length === 1 ? "residue" : "residues"}
          </p>
          <div className="flex flex-wrap gap-1">
            {residues.map(({ residue, kinds }) => (
              <Badge
                key={residue}
                variant={contactRank(kinds[0]) === 0 ? "secondary" : "outline"}
                className="gap-1.5"
              >
                {residue}
                <span className="font-normal text-muted-foreground">{kinds.join(" / ")}</span>
              </Badge>
            ))}
          </div>
        </div>
      ))}
      <p className="text-muted-foreground text-xs">
        Contacts are geometric: the pose is placed to make an interaction, which is not a
        measurement that it does.
      </p>
      <RowsTable rows={rows} columns={columns} total={total} />
    </div>
  );
}

function ModelFit({ data }: { data: Row }) {
  const meta = asRecord(data.meta);
  if (meta.ok === false) {
    return <p className="text-sm">{String(meta.reason ?? "The model could not be fitted.")}</p>;
  }
  const terms = isRows(data.terms) ? data.terms : [];
  const unmade = isRows(data.unmade) ? data.unmade : [];
  const reference = Object.entries(asRecord(meta.reference))
    .map(([position, group]) => `${position} ${String(group)}`)
    .join(", ");
  return (
    <div className="space-y-3">
      <Facts
        facts={[
          ["Compounds", meta.n_compounds],
          ["Excluded: censored activity values", meta.n_excluded_censored],
          // Zero of these is the usual case, and a 0 beside the censored count
          // reads as a second kind of exclusion the chemist has to think about.
          ["Excluded: missing substituent assignments", meta.n_excluded_incomplete || null],
          ["r²", meta.r2 != null ? cell(meta.r2) : null],
          ["q²", meta.q2 != null ? cell(meta.q2) : null],
          ["Reference", reference || null],
        ]}
      />
      <ForestChart
        data={terms.map((t) => ({
          label: `${String(t.position)} ${String(t.group)}`,
          value: typeof t.coef === "number" ? t.coef : 0,
          low: typeof t.ci_low === "number" ? t.ci_low : null,
          high: typeof t.ci_high === "number" ? t.ci_high : null,
        }))}
        xLabel="Change in activity from the reference group"
        height={Math.max(180, terms.length * 24 + 60)}
        caption="Each substituent against the reference at its position. Compare within a position, not across."
      />
      <RowsTable columns={["position", "group", "n", "coef"]} rows={terms} />
      {unmade.length > 0 && (
        <div className="space-y-1">
          <p className="text-sm text-muted-foreground">
            Predictions for unobserved substituent combinations
          </p>
          <RowsTable
            columns={["combination", "predicted"]}
            rows={unmade.map((u) => ({
              combination: Array.isArray(u.groups)
                ? u.groups.join(", ")
                : Object.entries(asRecord(u.groups))
                    .map(([position, group]) => `${position} ${String(group)}`)
                    .join(", "),
              predicted: u.prediction,
            }))}
          />
        </div>
      )}
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
  // Only the predictor sets ignored_ligands, so it still names the finder when no site came back.
  const scored =
    pockets.some((p) => "probability" in p) || "ignored_ligands" in meta
      ? "probability"
      : "druggability";
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
        {/* A predicted site is a patch of surface read from the protein atoms alone: what the
            cavity search says about cofactors and volumes does not apply to it. */}
        {scored === "probability"
          ? ignored.length > 0 && `Ignored by the predictor: ${ignored.join(", ")}.`
          : `${
              receptor.length > 0
                ? `${receptor.join(", ")} stayed in place as part of the receptor, so these are the cavities left beside ${receptor.length === 1 ? "it" : "them"}.`
                : "No cofactors were retained; reported pockets may therefore include cofactor-binding cavities."
            } Pocket volumes are estimated by Monte Carlo sampling and may vary between runs.`}
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
              ? `Show ${neighbours.length} related pocket${neighbours.length === 1 ? "" : "s"} in amber (rank${neighbours.length === 1 ? "" : "s"} ${neighbours.map((p) => cell(p.rank)).join(", ")})`
              : "No other pockets share lining residues or deposited ligand contacts with this pocket."}
          </label>
          <p className="text-xs text-muted-foreground">
            Deposited entry {structureId.toUpperCase()} with its ligands, and the lining residues of
            rank {cell(active.rank)} in magenta.{" "}
            {scored === "probability"
              ? "The predictor used only protein atoms from the prepared structure."
              : "fpocket ran on the prepared structure, which has no ligands or waters."}{" "}
            Select a table row to display another pocket.
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
                Select this pocket
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
  siblings = [],
}: {
  studyId: string;
  value: ValueSummary;
  onApprove?: (rank: number) => void;
  approveBlocked?: string | null;
  earlier?: boolean;
  /**
   * The other values the same step produced. Docking returns its poses and the receptor
   * it docked into as two values, and the pose is only worth drawing inside that
   * receptor -- so one needs to be able to find the other.
   */
  siblings?: ValueSummary[];
}) {
  const query = useStudyValue(studyId, value.id);
  const read = query.data;

  // Only fetched when this value is a set of poses; every other kind leaves it disabled.
  const receptorId =
    value.kind === "poses" ? siblings.find((v) => v.kind === "structure")?.id : undefined;
  // Fetched as a file: a value's own payload is served without its text.
  const receptorRead = useStudyValueFile(studyId, receptorId);
  const receptor = typeof receptorRead.data === "string" ? receptorRead.data : undefined;

  const columns = Array.isArray(read?.summary.columns)
    ? (read.summary.columns as string[])
    : isRows(read?.data) && read.data.length > 0
      ? Object.keys(read.data[0])
      : [];
  const total = typeof read?.summary.rows === "number" ? read.summary.rows : undefined;

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
  } else if (read.kind === "core_candidates") {
    body = (
      <Cores data={asRecord(read.data)} onApprove={onApprove} approveBlocked={approveBlocked} />
    );
  } else if (read.kind === "core") {
    body = <ApprovedCore data={asRecord(read.data)} earlier={earlier} />;
  } else if (read.kind === "model_fit") {
    body = <ModelFit data={asRecord(read.data)} />;
  } else if (read.kind === "poses" && isRows(read.data) && read.data.length > 0) {
    body = <Poses rows={read.data} columns={columns} total={total} receptor={receptor} />;
  } else if (read.kind === "interactions" && isRows(read.data) && read.data.length > 0) {
    body = <Interactions rows={read.data} columns={columns} total={total} />;
  } else if (read.kind === "effects" && isRows(read.data) && read.data.length > 0) {
    body = <Effects rows={read.data} columns={columns} total={total} />;
  } else if (read.kind === "rgroup_table" && isRows(read.data) && read.data.length > 0) {
    body = <RGroupTable rows={read.data} columns={columns} total={total} />;
  } else if (isRows(read.data) && read.data.length > 0) {
    // A substructure search says which query it ran, so the match can be coloured on every
    // structure rather than left as a column of true and false.
    const smarts = asRecord(read.summary.meta).smarts;
    const rows = read.data;
    // A table that carries a grouping carries the only answer its tool produced, and a
    // column of integers is not a form anyone can judge it in.
    const structure = columns.find((c) => STRUCTURE_COLUMN.test(c));
    const grouping = ["cluster", "scaffold_id"].find((c) => columns.includes(c));
    body = (
      <div className="space-y-3">
        {structure && grouping && (
          <GroupedStructures
            rows={rows}
            groupColumn={grouping}
            structureColumn={structure}
            centroidColumn={grouping === "cluster" ? "is_centroid" : undefined}
            label={grouping === "cluster" ? "Cluster" : "Scaffold"}
          />
        )}
        <RowsTable
          rows={rows}
          columns={columns}
          total={total}
          highlight={typeof smarts === "string" ? smarts : undefined}
        />
      </div>
    );
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
