"use client";

import { StructureThumbnail } from "@/shared/components/chemistry/structure-thumbnail";
import { studioGridTheme } from "@/shared/components/data-grid/ag-grid-theme";
import { ReadoutValue } from "@/shared/components/readout-value";
import { Button } from "@/shared/components/ui/button";
import { Label } from "@/shared/components/ui/label";
import { Switch } from "@/shared/components/ui/switch";
import type { ReadoutResponse } from "@/shared/lib/api/model";
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
import { fetchResultBlock } from "../hooks/use-runs";
import { IN_DOMAIN_FLOOR, buildResultParams } from "../lib/result-query";
import type { TriageRow } from "../types";

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

function StructureCell({ value }: { value: string }) {
  return <StructureThumbnail smiles={value} size={64} className="my-1" />;
}

/** A blank identifier, or a run from before identifiers were kept, is absence, not zero. */
function OrDash({ value }: { value: string | number | null }) {
  return value == null ? <span className="text-muted-foreground">—</span> : <span>{value}</span>;
}

export function TriageGrid({
  runId,
  readouts,
  onSaveSelection,
  saving,
}: {
  runId: string;
  readouts: ReadoutResponse[];
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

  const columns = useMemo<ColDef<TriageRow>[]>(() => {
    // No column for `__rowId`. AG Grid renders its own checkbox column from
    // `rowSelection.checkboxes`, and `row_id` is an internal handle for
    // `POST /collections` -- surfacing it would put a bare index in front of a
    // chemist, which is the same mistake as showing a UUID.
    const base: ColDef<TriageRow>[] = [
      {
        headerName: "Structure",
        field: "structure",
        width: 110,
        sortable: false,
        filter: false,
        cellRenderer: StructureCell,
      },
      {
        headerName: "SMILES",
        field: "structure",
        flex: 1,
        minWidth: 200,
        sortable: false,
        filter: false,
        cellClass: "font-mono text-xs",
      },
      // The join back to the scientist's own file. Neither is a sort key the
      // results endpoint accepts, so both are display only.
      {
        headerName: "ID",
        field: "compound_id",
        width: 140,
        sortable: false,
        filter: false,
        hide: !hasIds,
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
        headerTooltip: "Row number in the uploaded file, excluding the header",
        cellRenderer: OrDash,
      },
    ];

    // One column per Readout the Protocol declares, each rendered with its own
    // unit so a predicted value reads the way a measured one does.
    for (const readout of readouts) {
      base.push({
        headerName: readout.unit ? `${readout.name} (${readout.unit})` : readout.name,
        // A valueGetter column has no `field` to derive a colId from, and the
        // colId is what the API receives as the column name to sort by.
        colId: readout.name,
        width: 150,
        ...NUMBER_FILTER,
        valueGetter: (params) => params.data?.readouts?.[readout.name]?.value ?? null,
        cellRenderer: (params: { value: number | null }) => (
          <ReadoutValue value={params.value} unit={readout.unit} precision={3} />
        ),
      });
    }

    base.push(
      {
        headerName: "Uncertainty",
        field: "uncertainty",
        width: 130,
        ...NUMBER_FILTER,
        // Null for XGBoost, which has no ensemble spread to report. Rendered as
        // absence rather than as a fabricated zero.
        cellRenderer: (params: { value: number | null }) => (
          <ReadoutValue value={params.value} precision={3} />
        ),
      },
      {
        headerName: "Applicability",
        field: "applicability",
        width: 140,
        ...NUMBER_FILTER,
        cellRenderer: (params: { value: number | null }) =>
          params.value == null ? (
            <span className="text-muted-foreground">—</span>
          ) : (
            <span className={params.value < IN_DOMAIN_FLOOR ? "text-warning" : undefined}>
              {(params.value * 100).toFixed(0)}%
            </span>
          ),
      },
    );

    return base;
  }, [readouts, hasIds]);

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
