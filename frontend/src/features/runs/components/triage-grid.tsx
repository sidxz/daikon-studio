"use client";

import { StructureThumbnail } from "@/shared/components/chemistry/structure-thumbnail";
import { resultsGridTheme } from "@/shared/components/data-grid/ag-grid-theme";
import { Button } from "@/shared/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/shared/components/ui/dropdown-menu";
import { Label } from "@/shared/components/ui/label";
import { Switch } from "@/shared/components/ui/switch";
import { fileName } from "@/shared/lib/api/download";
import type { ReadoutResponse } from "@/shared/lib/api/model";
import { showError } from "@/shared/lib/toast";
import { cn } from "@/shared/lib/utils";
import {
  AllCommunityModule,
  type ColDef,
  type ColGroupDef,
  type GridApi,
  type IGetRowsParams,
  ModuleRegistry,
} from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
import { ChevronDown, Download, ListChecks, Maximize2, Minimize2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useResultsWorkspace } from "../hooks/use-results-workspace";
import { fetchResultBlock, useExportRunResults, useResultRanges } from "../hooks/use-runs";
import { positionIn } from "../lib/cell-scale";
import {
  filterLabel,
  readoutDescription,
  resultTargets,
  rowMatchesFilters,
  uncertaintyInfo,
} from "../lib/result-presentation";
import { IN_DOMAIN_FLOOR, buildResultParams } from "../lib/result-query";
import { EMPTY_VIEW, selectionKey, useColumnChoices, useResultViews } from "../lib/triage-state";
import type { TriageRow } from "../types";
import { ClassFilter } from "./class-filter";
import { CompoundSheet } from "./compound-sheet";
import {
  ApplicabilityCell,
  ClassCell,
  ProbabilityCell,
  UncertaintyCell,
  ValueCell,
} from "./result-cells";
import { resultHeader, resultTargetLabel } from "./result-column-header";
import { ResultColumnsMenu } from "./result-columns-menu";

ModuleRegistry.registerModules([AllCommunityModule]);
const BLOCK_SIZE = 100;
const NO_CHOICES: Record<string, boolean> = {};
const NUMBER_FILTER = {
  sortable: true,
  filter: "agNumberColumnFilter" as const,
  filterParams: {
    filterOptions: ["greaterThanOrEqual", "lessThanOrEqual", "inRange"],
    maxNumConditions: 1,
    buttons: ["reset"],
    inRangeInclusive: true,
  },
} satisfies Partial<ColDef<TriageRow>>;

export function columnHidden(id: string, shown: Record<string, boolean>, hasIds: boolean): boolean {
  if (id === "compound_id" && !hasIds) return true;
  return !(shown[id] ?? !["smiles", "input_row"].includes(id));
}

export function TriageGrid({
  runId,
  protocolId,
  engineId,
  readouts,
  totalCount,
  exportName,
  onSaveSelection,
  saving,
}: {
  runId: string;
  protocolId: string;
  engineId?: string;
  readouts: ReadoutResponse[];
  totalCount?: number;
  exportName: string;
  onSaveSelection: (rowIds: number[]) => void;
  saving: boolean;
}) {
  const { workspaceRef, height, maximized, setMaximized } = useResultsWorkspace();
  const apiRef = useRef<GridApi<TriageRow> | null>(null);
  const synchronizing = useRef(false);
  const bulkController = useRef<AbortController | null>(null);
  const bulkSnapshot = useRef("");
  const requestKey = useRef("");
  const view = useResultViews((state) => state.views[runId] ?? EMPTY_VIEW);
  const update = useResultViews((state) => state.update);
  const initialView = useRef(view);
  const shown = useColumnChoices((state) => state.protocols[protocolId] ?? NO_CHOICES);
  const setShown = useColumnChoices((state) => state.setShown);
  const { data: ranges } = useResultRanges(runId);
  const exporter = useExportRunResults();
  const [hasIds, setHasIds] = useState(false);
  const [inspected, setInspected] = useState<TriageRow | null>(null);
  const [loadedCount, setLoadedCount] = useState(0);
  const [matchingCount, setMatchingCount] = useState<number | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const [bulkCount, setBulkCount] = useState<number | null>(null);
  const targets = useMemo(() => resultTargets(readouts), [readouts]);
  const selected = Object.values(view.selected);
  const selectedIds = selected.map((row) => row.__rowId);
  const outsideCount = selected.filter(
    (row) => row.applicability != null && row.applicability < IN_DOMAIN_FLOOR,
  ).length;
  const hiddenCount = selected.filter(
    (row) => !rowMatchesFilters(row, view.filters, view.inDomainOnly, readouts),
  ).length;
  const filtered = view.inDomainOnly || Object.keys(view.filters).length > 0;
  const viewQuery = JSON.stringify(
    buildResultParams({
      sortModel: view.sort,
      filterModel: view.filters,
      inDomainOnly: view.inDomainOnly,
    }),
  );

  const syncSelection = useCallback(() => {
    const api = apiRef.current;
    if (!api || api.isDestroyed()) return;
    const picked = view.selected;
    synchronizing.current = true;
    let loaded = 0;
    api.forEachNode((node) => {
      if (!node.data) return;
      loaded++;
      const checked = Boolean(picked[selectionKey(node.data.__rowId)]);
      if (node.isSelected() !== checked) node.setSelected(checked);
    });
    synchronizing.current = false;
    setLoadedCount(loaded);
  }, [view.selected]);

  useEffect(() => {
    syncSelection();
  }, [syncSelection]);
  useEffect(() => {
    if (bulkSnapshot.current !== viewQuery) bulkController.current?.abort();
  }, [viewQuery]);
  useEffect(() => {
    // Bulk selection belongs to one filter/sort snapshot. A changed view or
    // unmount cancels it before any partial selection is committed.
    return () => {
      bulkController.current?.abort();
    };
  }, []);

  const toggleRow = useCallback(
    (row: TriageRow, checked: boolean) => {
      const picked = {
        ...(useResultViews.getState().views[runId]?.selected ?? {}),
      };
      if (checked) picked[selectionKey(row.__rowId)] = row;
      else delete picked[selectionKey(row.__rowId)];
      update(runId, { selected: picked });
    },
    [runId, update],
  );

  const { columns, choices } = useMemo(() => {
    const unavailable = (id: string) => ranges?.[id]?.min === null && ranges?.[id]?.max === null;
    const visible = (id: string) => !unavailable(id) && !columnHidden(id, shown, hasIds);
    const leaf = (id: string, definition: ColDef<TriageRow>): ColDef<TriageRow> => ({
      ...definition,
      colId: id,
      hide: !visible(id),
    });
    const identity: ColDef<TriageRow>[] = [
      leaf("structure", {
        ...resultHeader("structure", "2D molecule"),
        headerName: "Structure",
        field: "structure",
        width: 132,
        minWidth: 116,
        pinned: "left",
        lockPinned: true,
        cellRenderer: ({ data }: { data?: TriageRow }) =>
          data ? (
            <button
              type="button"
              aria-label={`Inspect ${data.compound_id ?? `compound at row ${data.input_row ?? data.__rowId + 1}`}`}
              className="rounded focus-visible:outline-2 focus-visible:outline-ring"
              onClick={() => setInspected(data)}
            >
              <StructureThumbnail smiles={data.structure} size={80} />
            </button>
          ) : null,
      }),
      leaf("compound_id", {
        ...resultHeader("id", "Compound ID"),
        headerName: "ID",
        field: "compound_id",
        width: 144,
        minWidth: 100,
        pinned: "left",
        lockPinned: true,
        cellRenderer: ({ data }: { data?: TriageRow }) =>
          data ? (
            <button
              type="button"
              className="truncate text-left underline-offset-4 hover:underline focus-visible:outline-2 focus-visible:outline-ring"
              onClick={() => setInspected(data)}
            >
              {data.compound_id ?? "—"}
            </button>
          ) : null,
      }),
      leaf("input_row", {
        ...resultHeader("row", "Original file"),
        headerName: "Input row",
        field: "input_row",
        width: 92,
        headerTooltip: "Row in the original compound list, excluding its header",
      }),
      leaf("smiles", {
        ...resultHeader("smiles", "Molecular notation"),
        headerName: "SMILES",
        field: "structure",
        width: 250,
        minWidth: 180,
        cellClass: "font-mono text-xs",
        cellRenderer: ({ value }: { value?: string }) => <span className="truncate">{value}</span>,
      }),
    ];
    const groups = targets.map((target) => {
      const cutoff = target.readout.threshold ?? 0.5;
      const outputs = [target.probability, target.readout].filter((r): r is ReadoutResponse =>
        Boolean(r),
      );
      const children = outputs.map((readout) => {
        const range = ranges?.[readout.name];
        const short =
          readout.type === "probability"
            ? "Probability"
            : readout.type === "class"
              ? "Class"
              : `Value${readout.unit ? ` (${readout.unit})` : ""}`;
        const header =
          targets.length > 1
            ? short
            : readout.type === "numeric"
              ? `${target.name}${readout.unit ? ` (${readout.unit})` : ""}`
              : `${target.name} · ${short}`;
        return leaf(readout.name, {
          ...resultHeader(
            readout.type ?? "numeric",
            readout.type === "class"
              ? "Predicted class"
              : readout.type === "probability"
                ? "Probability of class 1"
                : `Predicted value${readout.unit ? ` · ${readout.unit}` : ""}`,
            targets.length > 1
              ? readout.type === "probability"
                ? "Probability"
                : readout.type === "class"
                  ? "Class"
                  : "Value"
              : target.name,
          ),
          headerName: header,
          headerTooltip: `${readout.name}. ${readoutDescription(readout)}${readout.type === "probability" ? ` Classification cutoff: ${cutoff}.` : ""}`,
          width: readout.type === "class" ? 150 : 210,
          minWidth: readout.type === "class" ? 140 : 180,
          ...(readout.type === "class" ? {} : { flex: 1 }),
          ...NUMBER_FILTER,
          ...(readout.type === "class" ? { filter: ClassFilter, filterParams: undefined } : {}),
          valueGetter: ({ data }) => data?.readouts?.[readout.name]?.value ?? null,
          cellRenderer: ({ value }: { value: number | null }) =>
            readout.type === "probability" ? (
              <ProbabilityCell value={value} cutoff={cutoff} />
            ) : readout.type === "class" ? (
              <ClassCell value={value} />
            ) : (
              <ValueCell
                value={value}
                unit={readout.unit}
                fraction={value == null ? null : positionIn(value, range?.min, range?.max)}
                min={range?.min}
                max={range?.max}
              />
            ),
        });
      });
      const info = uncertaintyInfo(engineId, target.readout);
      const scale = info.ceiling ?? ranges?.[target.uncertainty]?.max ?? null;
      children.push(
        leaf(target.uncertainty, {
          ...resultHeader("uncertainty", "Model estimate"),
          headerName: "Uncertainty",
          headerTooltip: `${target.name}: ${info.description}`,
          width: 165,
          minWidth: 150,
          ...NUMBER_FILTER,
          valueGetter: ({ data }) => data?.uncertainty?.[target.name] ?? null,
          cellRenderer: ({ value }: { value: number | null }) => (
            <UncertaintyCell value={value} scale={scale} />
          ),
        }),
      );
      // Groups and columns share AG Grid's ID namespace. A group using the
      // readout name makes the leaf become `${name}_1`, which the API rejects.
      return {
        headerName: resultTargetLabel(target.name),
        headerClass: "results-header-group",
        groupId: `target:${target.name}`,
        marryChildren: true,
        children,
      };
    });
    const applicability = leaf("applicability", {
      ...resultHeader("applicability", "Training similarity"),
      headerName: "Applicability",
      field: "applicability",
      width: 185,
      minWidth: 165,
      ...NUMBER_FILTER,
      headerTooltip:
        "Tanimoto similarity to the nearest training compound. Inside the applicability domain at 30% or above. This is not prediction confidence.",
      cellRenderer: ({ value }: { value: number | null }) => <ApplicabilityCell value={value} />,
    });
    const choice = (column: ColDef<TriageRow>) => ({
      id: column.colId as string,
      label: column.headerName ?? "",
      shown: !column.hide,
      unavailable: unavailable(column.colId as string),
    });
    return {
      columns: [
        ...identity,
        ...(targets.length > 1 ? groups : groups.flatMap((g) => g.children)),
        applicability,
      ] as (ColDef<TriageRow> | ColGroupDef<TriageRow>)[],
      choices: [
        {
          id: "compound",
          name: "Compound",
          columns: identity.filter((c) => c.colId !== "compound_id" || hasIds).map(choice),
        },
        ...groups.map((g) => ({
          id: g.groupId,
          name: g.headerName,
          columns: g.children.map(choice),
        })),
        {
          id: "applicability",
          name: "Training similarity",
          columns: [choice(applicability)],
        },
      ],
    };
  }, [targets, ranges, shown, hasIds, engineId]);

  const datasource = useMemo(
    () => ({
      getRows: async (params: IGetRowsParams) => {
        const query = buildResultParams({
          sortModel: params.sortModel,
          filterModel: params.filterModel ?? {},
          inDomainOnly: view.inDomainOnly,
        });
        const key = JSON.stringify(query);
        if (requestKey.current !== key) {
          requestKey.current = key;
          setMatchingCount(null);
        }
        try {
          const { rows, nextCursor } = await fetchResultBlock(
            runId,
            params.startRow,
            params.endRow - params.startRow,
            query,
          );
          if (requestKey.current !== key) return;
          if (rows.some((row) => row.compound_id != null)) setHasIds(true);
          const lastRow = nextCursor === null ? params.startRow + rows.length : undefined;
          if (lastRow != null) setMatchingCount(lastRow);
          setLoadFailed(false);
          params.successCallback(rows, lastRow);
        } catch {
          if (requestKey.current !== key) return;
          setLoadFailed(true);
          params.failCallback();
        }
      },
    }),
    [runId, view.inDomainOnly],
  );

  const rememberView = useCallback(() => {
    const api = apiRef.current;
    if (!api) return;
    const sort = api
      .getColumnState()
      .filter((c) => c.sort)
      .map((c) => ({ colId: c.colId, sort: c.sort as "asc" | "desc" }));
    update(runId, { filters: api.getFilterModel(), sort, firstRow: 0 });
  }, [runId, update]);

  const selectLoaded = () => {
    const picked = { ...view.selected };
    apiRef.current?.forEachNode((node) => {
      if (node.data) picked[selectionKey(node.data.__rowId)] = node.data;
    });
    update(runId, { selected: picked });
  };

  const selectAllMatching = async () => {
    const controller = new AbortController();
    bulkController.current = controller;
    bulkSnapshot.current = viewQuery;
    setBulkCount(0);
    const rows: Record<string, TriageRow> = {};
    const query = buildResultParams({
      sortModel: view.sort,
      filterModel: view.filters,
      inDomainOnly: view.inDomainOnly,
    });
    let offset = 0;
    try {
      while (true) {
        const page = await fetchResultBlock(runId, offset, 500, query, controller.signal);
        if (controller.signal.aborted) return;
        for (const row of page.rows) rows[selectionKey(row.__rowId)] = row;
        setBulkCount(Object.keys(rows).length);
        if (page.nextCursor === null) break;
        const next = Number(page.nextCursor);
        if (!Number.isFinite(next) || next <= offset)
          throw new Error("Could not load the complete selection. Try again.");
        offset = next;
      }
      update(runId, {
        selected: {
          ...useResultViews.getState().views[runId]?.selected,
          ...rows,
        },
      });
    } catch (error) {
      if (!controller.signal.aborted)
        showError(error instanceof Error ? error.message : "Could not select matching compounds.");
    } finally {
      if (bulkController.current === controller) {
        bulkController.current = null;
        setBulkCount(null);
      }
    }
  };

  const exportRows = (rowIds?: number[]) =>
    exporter.mutate({
      runId,
      rowIds,
      params: buildResultParams({
        sortModel: view.sort,
        filterModel: view.filters,
        inDomainOnly: view.inDomainOnly,
      }),
      filename: fileName(`${exportName}${rowIds ? " selected" : ""}`, "xlsx", "predictions"),
    });

  return (
    <div ref={workspaceRef} className="relative h-[calc(100dvh-1rem)] min-h-80">
      <section
        aria-label="Prediction results table"
        className={cn(
          "flex min-h-0 flex-col gap-2 bg-background [&>div:not(.run-results-grid)]:shrink-0",
          maximized ? "fixed inset-0 z-40 p-3 sm:p-4" : "sticky top-2",
        )}
        style={maximized ? undefined : { height: height ?? "calc(100dvh - 18rem)" }}
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <p className="text-sm font-medium" aria-live="polite">
              {filtered
                ? matchingCount == null
                  ? "Filtered results"
                  : `${matchingCount.toLocaleString()} matching compounds`
                : `${(totalCount ?? matchingCount)?.toLocaleString() ?? "All"} compounds`}
            </p>
            <p className="text-xs text-muted-foreground">
              {targets.length} prediction target{targets.length === 1 ? "" : "s"} · Click a
              structure to inspect
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              aria-pressed={maximized}
              onClick={() => setMaximized(!maximized)}
              title={maximized ? "Restore page layout (Esc)" : "Use the full screen for results"}
            >
              {maximized ? <Minimize2 className="size-4" /> : <Maximize2 className="size-4" />}
              {maximized ? "Restore" : "Maximize"}
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={bulkCount !== null || loadedCount === 0}
                >
                  <ListChecks className="size-4" />
                  Select
                  <ChevronDown className="size-3.5" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onSelect={selectLoaded}>
                  Select loaded rows ({loadedCount})
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={() => void selectAllMatching()}>
                  Select all {filtered ? "matching" : "results"}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
            <ResultColumnsMenu
              groups={choices}
              onChange={(choices) => setShown(protocolId, choices)}
            />
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm" disabled={exporter.isPending}>
                  <Download className="size-4" />
                  {exporter.isPending ? "Exporting…" : "Export"}
                  <ChevronDown className="size-3.5" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onSelect={() => exportRows()}>
                  Export {filtered ? "filtered" : "all"} results to Excel
                </DropdownMenuItem>
                <DropdownMenuItem
                  disabled={!selected.length}
                  onSelect={() => exportRows(selectedIds)}
                >
                  Export selected ({selected.length}) to Excel
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2 rounded-lg border bg-muted/15 px-3 py-2">
          <div className="mr-2 flex items-center gap-2">
            <Switch
              id={`in-domain-${runId}`}
              checked={view.inDomainOnly}
              onCheckedChange={(inDomainOnly) => update(runId, { inDomainOnly, firstRow: 0 })}
            />
            <Label
              htmlFor={`in-domain-${runId}`}
              className="text-xs font-normal"
              title="Tanimoto similarity to the nearest training compound ≥ 0.3"
            >
              Within applicability domain
            </Label>
          </div>
          {Object.entries(view.filters).map(([column, model]) => (
            <Button
              key={column}
              size="sm"
              variant="secondary"
              className="h-7 max-w-full text-xs"
              onClick={() => {
                const filters = { ...view.filters };
                delete filters[column];
                apiRef.current?.setFilterModel(filters);
              }}
              aria-label={`Remove filter on ${column}`}
            >
              <span className="truncate">
                {column}{" "}
                {filterLabel(
                  model,
                  readouts.some((r) => r.name === column && r.type === "class"),
                )}
              </span>
              <X className="size-3" />
            </Button>
          ))}
          {filtered ? (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs"
              onClick={() => {
                apiRef.current?.setFilterModel(null);
                update(runId, { inDomainOnly: false, filters: {}, firstRow: 0 });
              }}
            >
              Clear filters
            </Button>
          ) : (
            <span className="text-xs text-muted-foreground">
              Use column menus to filter predictions.
            </span>
          )}
        </div>
        {bulkCount !== null && (
          <output className="flex items-center justify-between rounded-lg border px-3 py-2 text-sm">
            <span>Loading selection… {bulkCount.toLocaleString()} compounds</span>
            <Button variant="ghost" size="sm" onClick={() => bulkController.current?.abort()}>
              Cancel selection
            </Button>
          </output>
        )}
        {selected.length > 0 && (
          <div className="sticky top-0 z-20 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-primary/20 bg-background px-3 py-2 shadow-sm">
            <div className="text-sm" aria-live="polite">
              <span className="font-medium">{selected.length.toLocaleString()} selected</span>
              {hiddenCount > 0 && (
                <span className="text-muted-foreground">
                  {" "}
                  · {hiddenCount} hidden by current filters
                </span>
              )}
              {outsideCount > 0 && (
                <p className="text-xs text-warning">
                  {outsideCount} selected compounds outside the applicability domain
                </p>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Button size="sm" variant="ghost" onClick={() => update(runId, { selected: {} })}>
                Clear selection
              </Button>
              <Button size="sm" disabled={saving} onClick={() => onSaveSelection(selectedIds)}>
                {saving ? "Saving…" : `Save ${selected.length} as collection`}
              </Button>
            </div>
          </div>
        )}
        {loadFailed && (
          <div
            role="alert"
            className="flex items-center justify-between rounded-lg border border-destructive/30 px-3 py-2 text-sm"
          >
            <span>Could not load results.</span>
            <Button
              size="sm"
              variant="outline"
              onClick={() => apiRef.current?.refreshInfiniteCache()}
            >
              Try again
            </Button>
          </div>
        )}
        <div className="run-results-grid min-h-0 flex-1">
          <AgGridReact<TriageRow>
            theme={resultsGridTheme}
            columnDefs={columns}
            datasource={datasource}
            defaultColDef={{
              cellStyle: { display: "flex", alignItems: "center" },
              cellDataType: false,
              sortable: false,
              resizable: true,
              wrapHeaderText: true,
              autoHeaderHeight: true,
            }}
            suppressMultiSort
            hidePaddedHeaderRows
            tooltipShowDelay={250}
            rowModelType="infinite"
            cacheBlockSize={BLOCK_SIZE}
            maxBlocksInCache={10}
            infiniteInitialRowCount={Math.max(
              BLOCK_SIZE,
              initialView.current.firstRow + BLOCK_SIZE,
            )}
            rowHeight={92}
            headerHeight={60}
            groupHeaderHeight={32}
            rowSelection={{
              mode: "multiRow",
              checkboxes: true,
              headerCheckbox: false,
              enableClickSelection: false,
            }}
            selectionColumnDef={{
              pinned: "left",
              lockPinned: true,
              width: 44,
              resizable: false,
              cellStyle: { display: "flex", alignItems: "center" },
            }}
            getRowId={({ data }) => String(data.__rowId)}
            initialState={{
              filter: { filterModel: initialView.current.filters },
              sort: { sortModel: initialView.current.sort },
            }}
            onGridReady={({ api }) => {
              apiRef.current = api;
            }}
            onModelUpdated={syncSelection}
            onFirstDataRendered={({ api }) => {
              if (initialView.current.firstRow > 0)
                api.ensureIndexVisible(initialView.current.firstRow, "top");
            }}
            onBodyScrollEnd={({ api }) =>
              update(runId, {
                firstRow: Math.max(0, api.getFirstDisplayedRowIndex()),
              })
            }
            onFilterChanged={rememberView}
            onSortChanged={rememberView}
            onRowSelected={({ node, data }) => {
              if (data && !synchronizing.current) toggleRow(data, node.isSelected() === true);
            }}
            overlayNoRowsTemplate="<span>No compounds match these filters.</span>"
          />
        </div>
      </section>
      <CompoundSheet
        row={inspected}
        readouts={readouts}
        engineId={engineId}
        selected={inspected != null && Boolean(view.selected[selectionKey(inspected.__rowId)])}
        onClose={() => setInspected(null)}
        onSelect={() => {
          if (inspected) toggleRow(inspected, !view.selected[selectionKey(inspected.__rowId)]);
        }}
      />
    </div>
  );
}
