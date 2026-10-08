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

it("describes a split conflict as a partition disagreement, not as active/inactive", () => {
  // MoleculeACE is regression — there is no "active" anywhere in it. A compound that
  // appears on both sides of a published split is a leak, and the card has to say
  // which column caused it so the scientist knows what to fix.
  render(
    <ValidationReportView
      report={{
        total_rows: 6,
        valid_rows: 6,
        invalid: [],
        duplicates_collapsed: 0,
        salts_flagged: 0,
        duplicate_spread: {},
        conflicting: [
          { column: "split", structure: "CCO", values: ["test", "train"], row_numbers: [1, 2] },
        ],
      }}
    />,
  );
  expect(screen.queryByText(/active and inactive/)).not.toBeInTheDocument();
  expect(screen.getByText(/assigned to more than one partition/)).toBeInTheDocument();
  expect(screen.getByText("1 compound on both sides of the split")).toBeInTheDocument();
  expect(screen.getByText("Column")).toBeInTheDocument();
});
