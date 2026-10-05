import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { InitialsAvatar, initials } from "./initials-avatar";

const fallbackClass = (label: string) =>
  screen.getByRole("img", { name: label }).querySelector("[data-slot=avatar-fallback]")?.className;

describe("InitialsAvatar", () => {
  it("takes the first letters of the first and last words", () => {
    expect(initials("Ada Lovelace")).toBe("AL");
    expect(initials("Siddhant  K  Rath")).toBe("SR");
    expect(initials("Ada")).toBe("A");
  });

  it("tints two people with the same initials differently, and one person the same every time", () => {
    render(
      <>
        <InitialsAvatar name="Sid Rath" id="user-1" />
        <InitialsAvatar name="Siddhant Rath" id="user-2" />
      </>,
    );
    const sid = fallbackClass("Sid Rath");
    const siddhant = fallbackClass("Siddhant Rath");
    expect(screen.getByRole("img", { name: "Sid Rath" })).toHaveTextContent("SR");
    expect(screen.getByRole("img", { name: "Siddhant Rath" })).toHaveTextContent("SR");
    // The tint replaces the muted default, and the text size survives the merge.
    expect(sid).toMatch(/bg-(chart-\d|icon-collections)\/15/);
    expect(sid).toContain("text-[10px]");
    expect(sid).not.toContain("bg-muted");
    expect(sid).toContain("color-mix");
    expect(sid).not.toContain("text-muted-foreground");
    expect(sid).not.toEqual(siddhant);

    render(<InitialsAvatar name="Sid R." id="user-1" />);
    expect(fallbackClass("Sid R.")).toEqual(sid);
  });

  it("renders nothing for an unknown member", () => {
    const { container } = render(<InitialsAvatar name={undefined} id="user-1" />);
    expect(container).toBeEmptyDOMElement();
  });
});
