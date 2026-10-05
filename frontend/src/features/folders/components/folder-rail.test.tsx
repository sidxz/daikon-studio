import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ITEM_MIME } from "../lib/dnd";
import { FolderRail } from "./folder-rail";

const hoisted = vi.hoisted(() => ({
  canEdit: true,
  file: vi.fn(),
}));

vi.mock("../hooks/use-folders", () => {
  const mutation = () => ({ mutate: vi.fn(), mutateAsync: vi.fn() });
  return {
    useFolders: () => ({
      data: {
        can_edit: hoisted.canEdit,
        items: [
          { id: "f1", kind: "dataset", name: "hERG", item_count: 3, created_by: "u" },
          { id: "f2", kind: "dataset", name: "Solubility", item_count: 0, created_by: "u" },
        ],
      },
    }),
    useCreateFolder: mutation,
    useRenameFolder: mutation,
    useDeleteFolder: mutation,
    useFileItem: () => ({ mutate: hoisted.file }),
  };
});

function renderRail(onSelect = vi.fn()) {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <FolderRail kind="dataset" activeId="f1" onSelect={onSelect} />
    </QueryClientProvider>,
  );
  return onSelect;
}

function dataTransfer(kind: string, id: string) {
  const data: Record<string, string> = { [ITEM_MIME]: JSON.stringify({ kind, id }) };
  return {
    types: [ITEM_MIME, `${ITEM_MIME}.${kind}`],
    getData: (type: string) => data[type] ?? "",
  };
}

/** A card's drag, as the document sees it. jsdom has no DragEvent with a dataTransfer. */
function documentDrag(type: "dragstart" | "dragend", kind: string) {
  const event = new Event(type, { bubbles: true });
  Object.defineProperty(event, "dataTransfer", { value: dataTransfer(kind, "d1") });
  act(() => {
    document.dispatchEvent(event);
  });
}

describe("FolderRail", () => {
  beforeEach(() => {
    hoisted.canEdit = true;
    hoisted.file.mockReset();
  });

  it("renders All datasets and each folder with its count, and selects on click", () => {
    const onSelect = renderRail();
    const all = screen.getByRole("button", { name: "All datasets" });
    expect(all).toHaveAttribute("aria-pressed", "false");
    const herg = screen.getByRole("button", { name: /^hERG/ });
    expect(herg).toHaveAttribute("aria-pressed", "true");
    expect(herg).toHaveTextContent("3");
    expect(screen.getByTitle("Solubility")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /^Solubility/ }));
    expect(onSelect).toHaveBeenCalledWith("f2");
    fireEvent.click(all);
    expect(onSelect).toHaveBeenCalledWith(undefined);
  });

  it("offers folder actions and New folder to an editor", () => {
    renderRail();
    expect(screen.getByRole("button", { name: "New folder" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Folder actions for hERG" })).toBeInTheDocument();
  });

  it("has no New folder button or folder actions for a viewer", () => {
    hoisted.canEdit = false;
    renderRail();
    expect(screen.getByRole("button", { name: /^hERG/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /New folder/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Folder actions/ })).not.toBeInTheDocument();
  });

  it("files a dropped item of the same kind into the folder", () => {
    renderRail();
    fireEvent.drop(screen.getByTestId("folder-f2"), {
      dataTransfer: dataTransfer("dataset", "d1"),
    });
    expect(hoisted.file).toHaveBeenCalledWith({ itemId: "d1", folderId: "f2" });
  });

  it("ignores a drop of another kind", () => {
    renderRail();
    fireEvent.drop(screen.getByTestId("folder-f2"), {
      dataTransfer: dataTransfer("protocol", "p1"),
    });
    expect(hoisted.file).not.toHaveBeenCalled();
  });

  it("outlines every folder and changes the hint while a card of its kind is dragged", () => {
    renderRail();
    expect(screen.getByRole("heading", { name: "Folders" })).toBeInTheDocument();

    documentDrag("dragstart", "protocol");
    expect(screen.getByTestId("folder-f2").className).not.toContain("outline-dashed");

    documentDrag("dragstart", "dataset");
    expect(
      screen.getByRole("heading", { name: "Drop on a folder to file it" }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("folder-f1").className).toContain("outline-dashed");
    expect(screen.getByTestId("folder-f2").className).toContain("outline-dashed");

    documentDrag("dragend", "dataset");
    expect(screen.getByRole("heading", { name: "Folders" })).toBeInTheDocument();
    expect(screen.getByTestId("folder-f2").className).not.toContain("outline-dashed");
  });
});
