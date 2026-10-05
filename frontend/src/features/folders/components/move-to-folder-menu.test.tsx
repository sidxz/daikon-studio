import { customInstance } from "@/shared/lib/api/custom-instance";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MoveToFolderMenu } from "./move-to-folder-menu";

vi.mock("@/shared/lib/api/custom-instance", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/shared/lib/api/custom-instance")>()),
  customInstance: vi.fn(),
}));

let canEdit = true;

function renderMenu(currentFolderId: string | null) {
  vi.mocked(customInstance).mockImplementation(async ({ url, method }) => {
    if (method === "PUT") return {};
    if (url.endsWith("/folders")) {
      return {
        can_edit: canEdit,
        items: [{ id: "f1", kind: "protocol", name: "hERG", item_count: 1, created_by: "u" }],
      };
    }
    return {};
  });
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
    >
      <MoveToFolderMenu kind="protocol" itemId="p1" currentFolderId={currentFolderId} />
    </QueryClientProvider>,
  );
}

async function openMenu() {
  const trigger = await screen.findByRole("button", { name: "Move to folder" });
  fireEvent.keyDown(trigger, { key: "Enter" });
}

describe("MoveToFolderMenu", () => {
  beforeEach(() => {
    canEdit = true;
    vi.mocked(customInstance).mockReset();
  });

  it("files into the chosen folder", async () => {
    renderMenu(null);
    await openMenu();
    fireEvent.click(await screen.findByRole("menuitem", { name: "hERG" }));
    await waitFor(() =>
      expect(customInstance).toHaveBeenCalledWith(
        expect.objectContaining({
          url: "/api/v1/protocols/p1/folder",
          method: "PUT",
          data: { folder_id: "f1" },
        }),
      ),
    );
  });

  it("sends a null folder for No folder", async () => {
    renderMenu("f1");
    await openMenu();
    fireEvent.click(await screen.findByRole("menuitem", { name: "No folder" }));
    await waitFor(() =>
      expect(customInstance).toHaveBeenCalledWith(
        expect.objectContaining({ method: "PUT", data: { folder_id: null } }),
      ),
    );
  });

  it("is hidden from a viewer", async () => {
    canEdit = false;
    renderMenu(null);
    await waitFor(() => expect(customInstance).toHaveBeenCalled());
    expect(screen.queryByRole("button", { name: "Move to folder" })).not.toBeInTheDocument();
  });
});
