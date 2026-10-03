import { ApiError } from "@/shared/lib/api/custom-instance";
import { showError } from "@/shared/lib/toast";
import { useMutation } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { QueryProvider } from "./query-provider";

vi.mock("@/shared/lib/toast", () => ({ showError: vi.fn() }));

function Fail({ error, silent }: { error: Error; silent?: boolean }) {
  const { mutate, status } = useMutation({
    mutationFn: () => Promise.reject(error),
    meta: silent ? { silent } : undefined,
  });
  useEffect(() => mutate(), [mutate]);
  return <p>{status}</p>;
}

async function fail(error: Error, silent?: boolean) {
  render(
    <QueryProvider>
      <Fail error={error} silent={silent} />
    </QueryProvider>,
  );
  await screen.findByText("error");
}

describe("one toast per failed mutation", () => {
  afterEach(() => vi.mocked(showError).mockClear());

  it("toasts a failure nothing else handles", async () => {
    await fail(new Error("Could not save"));
    expect(showError).toHaveBeenCalledExactlyOnceWith("Could not save");
  });

  it("leaves a mutation marked silent to report its own failure", async () => {
    await fail(new Error("Could not save"), true);
    expect(showError).not.toHaveBeenCalled();
  });

  it("says nothing about a 401 while the session renews", async () => {
    await fail(new ApiError("Your session expired; signing you back in", 401, undefined, true));
    expect(showError).not.toHaveBeenCalled();
  });
});
