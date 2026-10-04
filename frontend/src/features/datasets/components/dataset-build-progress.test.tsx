import type { DatasetBuildResponse } from "@/shared/lib/api/model";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DatasetBuildProgress, buildPercent } from "./dataset-build-progress";

function build(overrides: Partial<DatasetBuildResponse> = {}): DatasetBuildResponse {
  return {
    id: "b1",
    name: "Nuisance",
    status: "running",
    stage: "Checking structures",
    done: 120_000,
    total: 403_993,
    dataset_id: null,
    error: null,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    ...overrides,
  };
}

describe("DatasetBuildProgress", () => {
  it("shows the stage, the percent and the structure count", () => {
    render(<DatasetBuildProgress build={build()} />);

    expect(screen.getByText("Checking structures")).toBeInTheDocument();
    expect(screen.getByText("29%")).toBeInTheDocument();
    expect(screen.getByText(/120,000 of 403,993 structures/)).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toBeInTheDocument();
  });

  it("shows no bar for a stage without a row count, rather than inventing one", () => {
    render(<DatasetBuildProgress build={build({ stage: "Saving", done: 0, total: 0 })} />);

    expect(screen.getByText("Saving")).toBeInTheDocument();
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
    expect(screen.queryByText(/structures/)).not.toBeInTheDocument();
  });

  it("never reports more than 100%", () => {
    expect(buildPercent({ done: 11, total: 10 })).toBe(100);
    expect(buildPercent({ done: 0, total: 0 })).toBeNull();
  });
});
