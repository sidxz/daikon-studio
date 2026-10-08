"use client";

import { type Editor, useEditorState } from "@tiptap/react";
import { Settings2 } from "lucide-react";
import type { ReactNode } from "react";

import { type PageSettings, pageSettings } from "@/features/pages/lib/page-settings";
import { Button } from "@/shared/components/ui/button";
import { Checkbox } from "@/shared/components/ui/checkbox";
import { Label } from "@/shared/components/ui/label";
import { Popover, PopoverContent, PopoverTrigger } from "@/shared/components/ui/popover";

/** Writes one settings attr onto the doc node. Goes through a transaction, so
 *  the editor's onUpdate fires and the normal dirty/Save flow persists it. */
function setDocAttr(editor: Editor, name: keyof PageSettings, value: string | boolean) {
  editor.view.dispatch(editor.state.tr.setDocAttribute(name, value));
}

function Choice({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <Button type="button" size="sm" variant={active ? "default" : "outline"} onClick={onClick}>
      {children}
    </Button>
  );
}

/** Layout popover in the editor toolbar: per-page settings, saved with the page
 *  (doc attributes — see lib/pages/page-settings.ts). Access lives in the
 *  sibling gear (PageAccessMenu). */
export function PageSettingsMenu({ editor }: { editor: Editor }) {
  const s = useEditorState({
    editor,
    selector: (ctx) =>
      pageSettings(ctx.editor ? { type: "doc", attrs: ctx.editor.state.doc.attrs } : null),
  });

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label="Page settings"
          title="Page settings"
          className="ml-auto"
        >
          <Settings2 />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-72 space-y-4">
        <div>
          <p className="mb-1.5 text-sm font-medium">Page width</p>
          <div className="flex gap-2">
            <Choice
              active={s.layout === "standard"}
              onClick={() => setDocAttr(editor, "layout", "standard")}
            >
              Standard
            </Choice>
            <Choice
              active={s.layout === "full"}
              onClick={() => setDocAttr(editor, "layout", "full")}
            >
              Full width
            </Choice>
          </div>
        </div>
        <div>
          <p className="mb-1.5 text-sm font-medium">Text size</p>
          <div className="flex gap-2">
            <Choice
              active={s.textSize === "default"}
              onClick={() => setDocAttr(editor, "textSize", "default")}
            >
              Default
            </Choice>
            <Choice
              active={s.textSize === "small"}
              onClick={() => setDocAttr(editor, "textSize", "small")}
            >
              Small
            </Choice>
          </div>
        </div>
        <div className="flex items-start gap-2">
          <Checkbox
            id="page-lock"
            checked={s.locked}
            onCheckedChange={(v) => setDocAttr(editor, "locked", v === true)}
          />
          <div className="space-y-0.5">
            <Label htmlFor="page-lock">Lock page</Label>
            <p className="text-xs text-muted-foreground">
              Prevents editing until unlocked. Takes effect after saving.
            </p>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}
