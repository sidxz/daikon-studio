import { describe, expect, it } from "vitest";

import { FigureImage } from "./figure-image";

describe("FigureImage node", () => {
  it("is a block node named figureImage with the expected attrs", () => {
    expect(FigureImage.name).toBe("figureImage");
    const attrs = FigureImage.config.addAttributes?.call(FigureImage as never) ?? {};
    expect(Object.keys(attrs)).toEqual(
      expect.arrayContaining(["blobKey", "mime", "caption", "alt"]),
    );
  });
});
