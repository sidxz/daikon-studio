import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const replace = vi.hoisted(() => vi.fn());
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace }) }));

const { default: PlansPage } = await import("./page");

describe("/plans", () => {
  it("sends old links to the new-study page without leaving the signed-in app", () => {
    render(<PlansPage />);
    expect(replace).toHaveBeenCalledWith("/studies/new");
  });
});
