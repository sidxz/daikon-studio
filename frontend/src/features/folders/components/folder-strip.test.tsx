import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ITEM_MIME } from "../lib/dnd";
import { FolderStrip } from "./folder-strip";

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

function renderStrip(onSelect = vi.fn()) {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <FolderStrip kind="dataset" activeId="f1" onSelect={onSelect} />
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

describe("FolderStrip", () => {
  beforeEach(() => {
    hoisted.canEdit = true;
    hoisted.file.mockReset();
  });

  it("renders All and each folder with its count, and selects on click", () => {
    const onSelect = renderStrip();
    expect(screen.getByRole("button", { name: "All" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("button", { name: /^hERG/ })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: /^hERG/ })).toHaveTextContent("3");
    fireEvent.click(screen.getByRole("button", { name: /^Solubility/ }));
    expect(onSelect).toHaveBeenCalledWith("f2");
    fireEvent.click(screen.getByRole("button", { name: "All" }));
    expect(onSelect).toHaveBeenCalledWith(undefined);
  });

  it("has no New folder button or tile menus for a viewer", () => {
    hoisted.canEdit = false;
    renderStrip();
    expect(screen.queryByRole("button", { name: /New folder/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Folder actions/ })).not.toBeInTheDocument();
  });

  it("files a dropped item of the same kind into the folder", () => {
    renderStrip();
    const tile = screen.getByTestId("folder-f2");
    fireEvent.drop(tile, { dataTransfer: dataTransfer("dataset", "d1") });
    expect(hoisted.file).toHaveBeenCalledWith({ itemId: "d1", folderId: "f2" });
  });

  it("ignores a drop of another kind", () => {
    renderStrip();
    fireEvent.drop(screen.getByTestId("folder-f2"), {
      dataTransfer: dataTransfer("protocol", "p1"),
    });
    expect(hoisted.file).not.toHaveBeenCalled();
  });
});
