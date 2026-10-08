import { describe, expect, it } from "vitest";

import { normalizeLinkHref } from "./editor-toolbar";

describe("normalizeLinkHref", () => {
  it("adds https:// to a scheme-less address so it isn't a relative in-app link", () => {
    expect(normalizeLinkHref("example.com")).toBe("https://example.com");
    expect(normalizeLinkHref("  example.com/a?b=1  ")).toBe("https://example.com/a?b=1");
  });

  it("leaves an existing scheme alone", () => {
    expect(normalizeLinkHref("https://example.com")).toBe("https://example.com");
    expect(normalizeLinkHref("HTTP://example.com")).toBe("HTTP://example.com");
    expect(normalizeLinkHref("mailto:me@example.com")).toBe("mailto:me@example.com");
    expect(normalizeLinkHref("ftp://files.example.com")).toBe("ftp://files.example.com");
  });

  it("does not disguise an unsafe scheme as a safe one — setLink still gets to refuse it", () => {
    expect(normalizeLinkHref("javascript:alert(1)")).toBe("javascript:alert(1)");
    expect(normalizeLinkHref("data:text/html,<script>")).toBe("data:text/html,<script>");
  });
});
