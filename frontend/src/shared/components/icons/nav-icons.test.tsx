import { navigation } from "@/shared/lib/navigation";
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

describe("navigation icons", () => {
  const items = navigation.flatMap((g) => g.items);
  it.each(items.map((i) => [i.title, i] as const))("%s renders a hidden, hued svg", (_, item) => {
    const Icon = item.icon;
    const { container } = render(<Icon className={item.iconClassName} />);
    const svg = container.querySelector("svg");
    expect(svg).not.toBeNull();
    expect(svg?.getAttribute("aria-hidden")).toBe("true");
    expect(item.iconClassName).toMatch(/^text-icon-/);
  });
});
