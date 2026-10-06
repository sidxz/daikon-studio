import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { ValidationReportView } from "./validation-report-view";

it("counts rows from a multi-target conflict once when reporting unique compounds", () => {
  render(
    <ValidationReportView
      report={{
        total_rows: 8,
        valid_rows: 8,
        invalid: [],
        duplicates_collapsed: 1,
        salts_flagged: 0,
        duplicate_spread: {},
        conflicting: [
          { column: "active", structure: "CCO", values: [0, 1], row_numbers: [1, 2, 3] },
          { column: "toxic", structure: "CCO", values: [0, 1], row_numbers: [1, 2, 3] },
        ],
      }}
    />,
  );
  const stat = screen.getByText("Unique compounds").parentElement;
  expect(stat).toHaveTextContent("4");
  expect(screen.getByText("1 compound with conflicting labels")).toBeInTheDocument();
});
