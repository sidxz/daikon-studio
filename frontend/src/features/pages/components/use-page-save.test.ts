// @vitest-environment jsdom
// renderHook mounts into a real DOM (document/window) — the rest of the suite
// runs under vitest's default "node" environment, so this is scoped to this
// file only rather than paying the jsdom cost repo-wide.
import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { usePageSave } from "./use-page-save";

describe("usePageSave", () => {
  it("never saves on change or unmount — only via saveNow", async () => {
    const save = vi.fn(async () => {});
    const { result, unmount } = renderHook(() => usePageSave({ save }));

    act(() => result.current.onChange({ type: "doc", content: [1] }));
    expect(result.current.dirty).toBe(true);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(save).not.toHaveBeenCalled();

    await act(() => result.current.saveNow());
    expect(save).toHaveBeenCalledWith({ type: "doc", content: [1] });
    expect(result.current.dirty).toBe(false);

    act(() => result.current.onChange({ type: "doc", content: [2] }));
    unmount();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(save).toHaveBeenCalledTimes(1); // unmount must not save silently
  });

  it("saves the latest doc when several edits precede one save", async () => {
    const save = vi.fn(async () => {});
    const { result } = renderHook(() => usePageSave({ save }));

    act(() => result.current.onChange({ type: "doc", content: [1] }));
    act(() => result.current.onChange({ type: "doc", content: [2] }));
    await act(() => result.current.saveNow());

    expect(save).toHaveBeenCalledTimes(1);
    expect(save).toHaveBeenCalledWith({ type: "doc", content: [2] });
  });

  it("saveNow with nothing pending is a no-op that still reports success", async () => {
    const save = vi.fn(async () => {});
    const { result } = renderHook(() => usePageSave({ save }));

    await act(async () => {
      expect(await result.current.saveNow()).toBe(true);
    });
    expect(save).not.toHaveBeenCalled();
  });

  it("surfaces a 409 as conflict state", async () => {
    const { ApiError } = await import("@/shared/lib/api/custom-instance");
    const save = vi.fn(async () => {
      throw new ApiError("conflict", 409, undefined);
    });
    const { result } = renderHook(() => usePageSave({ save }));

    act(() => result.current.onChange({ type: "doc" }));
    await act(() => result.current.saveNow());
    await waitFor(() => expect(result.current.conflict).toBe(true));
  });

  it("keeps the doc after a failed save so saveNow retries without a new edit", async () => {
    const save = vi
      .fn<(doc: unknown) => Promise<void>>()
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValueOnce(undefined);
    const { result } = renderHook(() => usePageSave({ save }));

    act(() => result.current.onChange({ type: "doc", content: [7] }));
    await act(() => result.current.saveNow());
    expect(result.current.status).toBe("error");
    expect(result.current.dirty).toBe(true);

    await act(() => result.current.saveNow());
    expect(result.current.status).toBe("saved");
    expect(result.current.dirty).toBe(false);
    expect(save).toHaveBeenLastCalledWith({ type: "doc", content: [7] });
  });

  it("discard drops the pending doc", async () => {
    const save = vi.fn(async () => {});
    const { result } = renderHook(() => usePageSave({ save }));

    act(() => result.current.onChange({ type: "doc", content: [3] }));
    act(() => result.current.discard());
    expect(result.current.dirty).toBe(false);

    await act(() => result.current.saveNow());
    expect(save).not.toHaveBeenCalled();
  });
});
