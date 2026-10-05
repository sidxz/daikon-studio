"use client";

import { StructureThumbnail } from "@/shared/components/chemistry/structure-thumbnail";
import { studioGridTheme } from "@/shared/components/data-grid/ag-grid-theme";
import { Button } from "@/shared/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/shared/components/ui/dropdown-menu";
import { Label } from "@/shared/components/ui/label";
import { Switch } from "@/shared/components/ui/switch";
import { fileName } from "@/shared/lib/api/download";
import type { ReadoutResponse } from "@/shared/lib/api/model";
import { targetsOf, uncertaintyColumn } from "@/shared/lib/targets";
import {
  AllCommunityModule,
  type ColDef,
  type GridApi,
  type GridReadyEvent,
  type IDatasource,
  ModuleRegistry,
} from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { fetchResultBlock, useExportRunResults, useResultRanges } from "../hooks/use-runs";
import { PROBABILITY_SPREAD_MAX, cutoffFor, positionIn } from "../lib/cell-scale";
import { IN_DOMAIN_FLOOR, buildResultParams } from "../lib/result-query";
import type { TriageRow } from "../types";
import {
  ApplicabilityCell,
  ClassCell,
  ProbabilityCell,
  UncertaintyCell,
  ValueCell,
} from "./result-cells";

ModuleRegistry.registerModules([AllCommunityModule]);

const BLOCK_SIZE = 100;

const NUMBER_FILTER = {
  sortable: true,
  filter: "agNumberColumnFilter" as const,
  filterParams: {
    // Only what the API can honour. Offering "not equal" or "blank" would be
    // a control that quietly filters by something else.
    filterOptions: ["greaterThanOrEqual", "lessThanOrEqual", "inRange"],
    maxNumConditions: 1,
    buttons: ["reset"],
    // The server applies `>=` / `<=` -- inclusive on both ends. AG Grid's own
    // default for "in range" is exclusive of the upper bound, which would show
    // a control whose semantics disagree with what actually gets sent.
    inRangeInclusive: true,
  },
} satisfies Partial<ColDef<TriageRow>>;

const CENTRED = { display: "flex", alignItems: "center" };

/** Hidden until asked for: the structure drawing already shows the molecule. */
const HIDDEN_BY_DEFAULT = new Set(["smiles"]);

/** Which grid columns this browser has chosen to show or hide, by column id; one
 * choice for every run, so hiding SMILES once hides it everywhere. */
const useColumnChoices = create<{
  shown: Record<string, boolean>;
  setShown: (column: string, shown: boolean) => void;
}>()(
  persist(
    (set) => ({
      shown: {},
      setShown: (column, shown) => set((state) => ({ shown: { ...state.shown, [column]: shown } })),
    }),
    { name: "ds-triage-columns" },
  ),
);

function SmilesCell({ value }: { value: string }) {
  return <span className="min-w-0 truncate">{value}</span>;
}

function StructureCell({ value }: { value: string }) {
  return <StructureThumbnail smiles={value} size={64} className="my-1" />;
}

/** A blank identifier, or a run from before identifiers were kept, is absence, not zero. */
function OrDash({ value }: { value: string | number | null }) {
  return value == null ? <span className="text-muted-foreground">—</span> : <span>{value}</span>;
}

/** This browser's choice for the column, else its default. The ID column stays hidden
 * for an upload that had no identifier column, whatever was chosen: it would be empty. */
export function columnHidden(id: string, shown: Record<string, boolean>, hasIds: boolean): boolean {
  if (id === "compound_id" && !hasIds) return true;
  return !(shown[id] ?? !HIDDEN_BY_DEFAULT.has(id));
}

/** A column's id: its own `colId`, or the field AG Grid derives one from. */
function columnId(column: ColDef<TriageRow>): string {
  return column.colId ?? String(column.field);
}

export function TriageGrid({
  runId,
  readouts,
  exportName,
  onSaveSelection,
  saving,
}: {
  runId: string;
  readouts: ReadoutResponse[];
  /** What the downloaded workbook is named after, e.g. the protocol and date. */
  exportName: string;
  onSaveSelection: (rowIds: number[]) => void;
  saving: boolean;
}) {
  const apiRef = useRef<GridApi<TriageRow> | null>(null);
  const [selected, setSelected] = useState<number[]>([]);
  const [outsideDomain, setOutsideDomain] = useState(0);
  const [inDomainOnly, setInDomainOnly] = useState(false);
  // Unknown until a block arrives: the Run does not say whether it was given an
  // identifier column. Set from any block, so leading blank IDs cannot hide it.
  const [hasIds, setHasIds] = useState(false);
  // The scale for each column's bars: the whole run, not the page in view. Until it
  // arrives a continuous column shows its numbers alone.
  const { data: ranges } = useResultRanges(runId);
  const { shown, setShown } = useColumnChoices();
  const exporter = useExportRunResults();

  // The grid's own sort, filters and domain switch, exactly as its pages request
  // them, so the file holds what the grid shows.
  const exportView = useCallback(() => {
    const api = apiRef.current;
    if (!api) return;
    const sortModel = api
      .getColumnState()
      .filter((column) => column.sort)
      .sort((a, b) => (a.sortIndex ?? 0) - (b.sortIndex ?? 0))
      .map((column) => ({ colId: column.colId, sort: column.sort as "asc" | "desc" }));
    exporter.mutate({
      runId,
      params: buildResultParams({ sortModel, filterModel: api.getFilterModel(), inDomainOnly }),
      filename: fileName(exportName, "xlsx", "predictions"),
    });
  }, [exporter, runId, inDomainOnly, exportName]);

  const columns = useMemo<ColDef<TriageRow>[]>(() => {
    // No column for `__rowId`. AG Grid renders its own checkbox column from
    // `rowSelection.checkboxes`, and `row_id` is an internal handle for
    // `POST /collections` -- surfacing it would put a bare index in front of a
    // chemist, which is the same mistake as showing a UUID.
    const base: ColDef<TriageRow>[] = [
      {
        headerName: "Structure",
        colId: "structure",
        field: "structure",
        width: 110,
        sortable: false,
        filter: false,
        cellRenderer: StructureCell,
      },
      {
        headerName: "SMILES",
        colId: "smiles",
        field: "structure",
        flex: 2,
        minWidth: 200,
        sortable: false,
        filter: false,
        cellClass: "font-mono text-xs",
        // Its own span: a bare text node in a flex cell loses its ellipsis.
        cellRenderer: SmilesCell,
      },
      // The join back to the scientist's own file. Neither is a sort key the
      // results endpoint accepts, so both are display only.
      {
        headerName: "ID",
        field: "compound_id",
        width: 140,
        sortable: false,
        filter: false,
        cellRenderer: OrDash,
      },
      {
        headerName: "Row",
        field: "input_row",
        width: 80,
        sortable: false,
        filter: false,
        // `input_row` counts data rows from 1, as the upload's ValidationReport
        // does; the CSV's own line number is one more, for the header.
        headerTooltip: "Row number in the compound list, excluding any header",
        cellRenderer: OrDash,
      },
    ];

    // One column per Readout the Protocol declares, each rendered with its own
    // unit so a predicted value reads the way a measured one does, and drawn by its
    // type: a probability against its cutoff, a class as a word, a continuous value
    // on the run's own range.
    for (const readout of readouts) {
      const range = ranges?.[readout.name];
      const cutoff = cutoffFor(readout.name, readouts);
      base.push({
        headerName: readout.unit ? `${readout.name} (${readout.unit})` : readout.name,
        // A valueGetter column has no `field` to derive a colId from, and the
        // colId is what the API receives as the column name to sort by.
        colId: readout.name,
        width: 160,
        ...NUMBER_FILTER,
        valueGetter: (params) => params.data?.readouts?.[readout.name]?.value ?? null,
        cellRenderer: (params: { value: number | null }) =>
          readout.type === "probability" ? (
            <ProbabilityCell value={params.value} cutoff={cutoff} />
          ) : readout.type === "class" ? (
            <ClassCell value={params.value} />
          ) : (
            <ValueCell
              value={params.value}
              unit={readout.unit}
              fraction={
                params.value == null ? null : positionIn(params.value, range?.min, range?.max)
              }
              min={range?.min}
              max={range?.max}
            />
          ),
      });
    }

    // One per target: each target's own model reports its own spread. The colId is
    // the results column the API sorts and filters by -- plain `uncertainty` for a
    // one-target Protocol, as every results file before several targets used.
    const targets = targetsOf(readouts);
    for (const target of targets) {
      const column = uncertaintyColumn(target, targets.length);
      const runMax = ranges?.[column]?.max ?? null;
      // A binary target's uncertainty is drawn on at least 0 to 0.5, the most an
      // ensemble's spread of probabilities can be, so a run of confident predictions
      // does not fill its bars. Engines that report distance from the boundary
      // instead reach 1, which the run's own maximum then covers. A continuous
      // target has no such ceiling, so its scale is the run's own maximum.
      const isBinary = readouts.some((r) => r.name === target && r.type === "class");
      const scale =
        runMax == null ? null : isBinary ? Math.max(PROBABILITY_SPREAD_MAX, runMax) : runMax;
      base.push({
        headerName: targets.length === 1 ? "Uncertainty" : `Uncertainty (${target})`,
        colId: column,
        width: 150,
        ...NUMBER_FILTER,
        valueGetter: (params) => params.data?.uncertainty?.[target] ?? null,
        // Null for XGBoost, which has no ensemble spread to report. Rendered as
        // absence rather than as a fabricated zero.
        cellRenderer: (params: { value: number | null }) => (
          <UncertaintyCell value={params.value} scale={scale} />
        ),
      });
    }

    base.push({
      headerName: "Applicability",
      field: "applicability",
      width: 140,
      ...NUMBER_FILTER,
      cellRenderer: (params: { value: number | null }) => (
        <ApplicabilityCell value={params.value} />
      ),
    });

    // Every column but the drawing shares the spare width, each at least as wide as
    // before, so hiding one (SMILES used to take all of it) leaves no blank strip.
    return base.map((column) => ({
      ...column,
      hide: columnHidden(columnId(column), shown, hasIds),
      flex: column.colId === "structure" ? undefined : (column.flex ?? 1),
      minWidth: column.minWidth ?? column.width,
    }));
  }, [readouts, hasIds, ranges, shown]);

  const datasource = useMemo<IDatasource>(
    () => ({
      rowCount: undefined,
      getRows: async (params) => {
        try {
          const { rows, nextCursor } = await fetchResultBlock(
            runId,
            params.startRow,
            params.endRow - params.startRow,
            buildResultParams({
              sortModel: params.sortModel,
              filterModel: params.filterModel ?? {},
              inDomainOnly,
            }),
          );
          if (rows.some((row) => row.compound_id != null)) setHasIds(true);
          // `total_count` is always null in this API, so the last row is only
          // known when a page comes back without a next cursor.
          const lastRow = nextCursor === null ? params.startRow + rows.length : undefined;
          params.successCallback(rows, lastRow);
        } catch {
          params.failCallback();
        }
      },
    }),
    [runId, inDomainOnly],
  );

  const onGridReady = useCallback(
    (event: GridReadyEvent<TriageRow>) => {
      apiRef.current = event.api;
      event.api.setGridOption("datasource", datasource);
    },
    [datasource],
  );

  useEffect(() => {
    // AG Grid purges its block cache when the sort or filter model changes,
    // but the in-domain switch lives outside both -- re-setting the datasource
    // is what makes it restart from offset 0 instead of appending a filtered
    // page onto unfiltered blocks.
    apiRef.current?.setGridOption("datasource", datasource);
  }, [datasource]);

  const refreshSelection = useCallback(() => {
    const api = apiRef.current;
    if (!api) return;
    const rows = api.getSelectedRows();
    setSelected(rows.map((row) => row.__rowId));
    setOutsideDomain(
      rows.filter((row) => row.applicability != null && row.applicability < IN_DOMAIN_FLOOR).length,
    );
  }, []);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <Switch id="in-domain" checked={inDomainOnly} onCheckedChange={setInDomainOnly} />
          <Label htmlFor="in-domain" className="text-sm font-normal">
            Within applicability domain
          </Label>
        </div>
        <div className="ml-auto flex items-center gap-3">
          {selected.length > 0 && outsideDomain > 0 && (
            // A forecast about the selection, so it sits beside the button
            // rather than inside it. This one is the whole point of the
            // applicability column: better decisions come from knowing how
            // many of your picks the model has never seen anything like.
            <span className="text-sm text-warning">
              {outsideDomain} of {selected.length} selected compounds are outside the applicability
              domain
            </span>
          )}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">Columns</Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {columns
                .filter((column) => columnId(column) !== "compound_id" || hasIds)
                .map((column) => (
                  <DropdownMenuCheckboxItem
                    key={columnId(column)}
                    checked={!column.hide}
                    onCheckedChange={(checked) => setShown(columnId(column), checked === true)}
                    // Stays open, so several columns can be switched in one go.
                    onSelect={(event) => event.preventDefault()}
                  >
                    {column.headerName}
                  </DropdownMenuCheckboxItem>
                ))}
            </DropdownMenuContent>
          </DropdownMenu>
          <Button variant="outline" disabled={exporter.isPending} onClick={exportView}>
            {exporter.isPending ? "Exporting…" : "Export to Excel"}
          </Button>
          <Button
            disabled={selected.length === 0 || saving}
            onClick={() => onSaveSelection(selected)}
          >
            {saving
              ? "Saving…"
              : selected.length === 0
                ? "Save as collection"
                : `Save ${selected.length} as collection`}
          </Button>
        </div>
      </div>

      {/* An explicit height. A percentage through a flex parent is not a
          definite height, and the grid then renders every row it has instead
          of windowing. */}
      <div style={{ height: "calc(100vh - 20rem)", minHeight: 420 }}>
        <AgGridReact<TriageRow>
          theme={studioGridTheme}
          columnDefs={columns}
          // Every cell centred in the tall rows the structure thumbnails need, so a
          // number, a bar and a label line up across the row. An inline style, not a
          // class: AG Grid's stylesheet outranks Tailwind's layered utilities.
          defaultColDef={{ cellStyle: CENTRED }}
          rowModelType="infinite"
          cacheBlockSize={BLOCK_SIZE}
          rowHeight={72}
          rowSelection={{
            mode: "multiRow",
            checkboxes: true,
            headerCheckbox: false,
            enableClickSelection: false,
          }}
          // The server's row_id: the row's position in the original results
          // file, minted before any sort or filter. That's exactly what
          // POST /collections wants back as row_ids, and what keeps a
          // selection alive across blocks AG Grid has already discarded --
          // a page offset would disagree with it under any sort or filter.
          getRowId={(params) => String(params.data.__rowId)}
          onGridReady={onGridReady}
          onSelectionChanged={refreshSelection}
        />
      </div>
    </div>
  );
}
