"use client";

import { StructureThumbnail } from "@/shared/components/chemistry/structure-thumbnail";
import { studioGridTheme } from "@/shared/components/data-grid/ag-grid-theme";
import { ReadoutValue } from "@/shared/components/readout-value";
import { Button } from "@/shared/components/ui/button";
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
import { useCallback, useMemo, useRef, useState } from "react";
import { fetchResultBlock } from "../hooks/use-runs";
import type { TriageRow } from "../types";

ModuleRegistry.registerModules([AllCommunityModule]);

const BLOCK_SIZE = 100;

function StructureCell({ value }: { value: string }) {
  return <StructureThumbnail smiles={value} size={64} className="my-1" />;
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

  const columns = useMemo<ColDef<TriageRow>[]>(() => {
    // No column for `__rowId`. AG Grid renders its own checkbox column from
    // `rowSelection.checkboxes`, and the offset is an internal handle for
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
        cellClass: "font-mono text-xs",
      },
    ];

    // One column per Readout the Protocol declares, each rendered with its own
    // unit so a predicted value reads the way a measured one does.
    for (const readout of readouts) {
      base.push({
        headerName: readout.unit ? `${readout.name} (${readout.unit})` : readout.name,
        width: 150,
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
        cellRenderer: (params: { value: number | null }) =>
          params.value == null ? (
            <span className="text-muted-foreground">—</span>
          ) : (
            <span className={params.value < 0.5 ? "text-warning" : undefined}>
              {(params.value * 100).toFixed(0)}%
            </span>
          ),
      },
    );

    return base;
  }, [readouts]);

  const datasource = useMemo<IDatasource>(
    () => ({
      rowCount: undefined,
      getRows: async (params) => {
        try {
          const { rows, nextCursor } = await fetchResultBlock(
            runId,
            params.startRow,
            params.endRow - params.startRow,
          );
          // `total_count` is always null in this API, so the last row is only
          // known when a page comes back without a next cursor.
          const lastRow = nextCursor === null ? params.startRow + rows.length : undefined;
          params.successCallback(rows, lastRow);
        } catch {
          params.failCallback();
        }
      },
    }),
    [runId],
  );

  const onGridReady = useCallback(
    (event: GridReadyEvent<TriageRow>) => {
      apiRef.current = event.api;
      event.api.setGridOption("datasource", datasource);
    },
    [datasource],
  );

  const refreshSelection = useCallback(() => {
    const api = apiRef.current;
    if (!api) return;
    const rows = api.getSelectedRows();
    setSelected(rows.map((row) => row.__rowId));
    setOutsideDomain(
      rows.filter((row) => row.applicability != null && row.applicability < 0.5).length,
    );
  }, []);

  return (
    <div className="space-y-3">
      {/* No filter box. AG Grid's quick filter is client-side only and this
          grid is infinite, and `GET /runs/{id}/results` takes no filter
          parameter -- so a search input here would be a control that silently
          does nothing, which is worse than not offering one. Sorting and
          filtering the results server-side is a real gap, recorded as such. */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="ml-auto flex items-center gap-3">
          {selected.length > 0 && outsideDomain > 0 && (
            // A forecast about the selection, so it sits beside the button
            // rather than inside it. This one is the whole point of the
            // applicability column: better decisions come from knowing how
            // many of your picks the model has never seen anything like.
            <span className="text-sm text-warning">
              {outsideDomain} of your {selected.length} are outside the domain of applicability
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
          // Absolute offset, which is exactly what POST /collections wants back
          // as row_ids -- and what keeps a selection alive across blocks AG Grid
          // has already discarded.
          getRowId={(params) => String(params.data.__rowId)}
          onGridReady={onGridReady}
          onSelectionChanged={refreshSelection}
        />
      </div>
    </div>
  );
}
