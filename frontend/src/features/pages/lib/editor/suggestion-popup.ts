import { ReactRenderer } from "@tiptap/react";
import type { SuggestionOptions } from "@tiptap/suggestion";
import type {
  ForwardRefExoticComponent,
  ForwardedRef,
  PropsWithoutRef,
  RefAttributes,
} from "react";
import { useImperativeHandle, useState } from "react";

/** What every suggestion list exposes to its popup: the popup forwards raw
 *  keydowns and the list answers whether it consumed them. */
export type SuggestionListRef = { onKeyDown: (e: KeyboardEvent) => boolean };

const POPUP_Z_INDEX = 60; // above the card dialog (overlays are z-50)
const VIEWPORT_MARGIN = 8; // breathing room between the popup and the viewport edge
const FLIP_MIN_HEIGHT = 160; // flip above the caret when less than this fits below

/** Callback ref for the active row of a suggestion list: the popup caps its
 *  height to the viewport and scrolls, so arrow-key moves must keep the active
 *  row visible. Stable identity — React re-invokes it only when a row gains or
 *  loses it. */
export const scrollActiveIntoView = (el: HTMLElement | null) =>
  el?.scrollIntoView({ block: "nearest" });

type ListComponent<TItem, TSelected> = ForwardRefExoticComponent<
  PropsWithoutRef<{ items: TItem[]; command: (item: TSelected) => void }> &
    RefAttributes<SuggestionListRef>
>;

/**
 * The `render` half of a `@tiptap/suggestion` channel: mounts `List` in a
 * fixed-position popup at the caret, keeps it there while the page scrolls or
 * resizes, and tears it down on Escape/exit. Shared by every channel — `@`
 * people and `#` entities (lib/editor/mention.ts) and `/` commands
 * (components/pages/slash-menu.ts) — which differ only in the list they show.
 */
export function suggestionPopup<TItem, TSelected>(
  List: ListComponent<TItem, TSelected>,
): NonNullable<SuggestionOptions<TItem, TSelected>["render"]> {
  return () => {
    let renderer: ReactRenderer<SuggestionListRef> | null = null;
    let popup: HTMLDivElement | null = null;
    // Latest caret-rect getter; kept fresh so scroll/resize can reposition
    // between tiptap updates.
    let getRect: (() => DOMRect | null) | null | undefined;

    // Keeps the popup inside the viewport: height is capped to the space on the
    // chosen side (the list scrolls internally), it flips above the caret when
    // the space below is too cramped, and the left edge clamps so a caret near
    // the right edge doesn't push it off-screen.
    const place = (rect: DOMRect | null | undefined) => {
      if (!popup || !rect) return;
      const below = window.innerHeight - rect.bottom - 4 - VIEWPORT_MARGIN;
      if (below >= FLIP_MIN_HEIGHT) {
        popup.style.top = `${rect.bottom + 4}px`;
        popup.style.bottom = "";
        popup.style.maxHeight = `${below}px`;
      } else {
        popup.style.top = "";
        popup.style.bottom = `${window.innerHeight - rect.top + 4}px`;
        popup.style.maxHeight = `${rect.top - 4 - VIEWPORT_MARGIN}px`;
      }
      const width = popup.offsetWidth;
      popup.style.left = `${Math.max(
        VIEWPORT_MARGIN,
        Math.min(rect.left, window.innerWidth - width - VIEWPORT_MARGIN),
      )}px`;
    };

    const reposition = () => place(getRect?.());

    const teardown = () => {
      window.removeEventListener("scroll", reposition, true);
      window.removeEventListener("resize", reposition);
      popup?.remove();
      renderer?.destroy();
      renderer = null;
      popup = null;
      getRect = undefined;
    };

    return {
      onStart: (props) => {
        getRect = props.clientRect;
        renderer = new ReactRenderer(List, { props, editor: props.editor });
        popup = document.createElement("div");
        popup.style.position = "fixed";
        popup.style.zIndex = String(POPUP_Z_INDEX);
        popup.style.overflowY = "auto";
        popup.appendChild(renderer.element);
        document.body.appendChild(popup);
        // Capture-phase scroll catches scrolling in any ancestor (the card
        // dialog scrolls its own body); resize covers viewport changes.
        window.addEventListener("scroll", reposition, true);
        window.addEventListener("resize", reposition);
        place(getRect?.());
      },
      onUpdate: (props) => {
        if (!renderer || !popup) return;
        getRect = props.clientRect;
        renderer.updateProps(props);
        place(getRect?.());
      },
      onKeyDown: (props) => {
        if (props.event.key === "Escape") {
          teardown();
          return true;
        }
        return renderer?.ref?.onKeyDown(props.event) ?? false;
      },
      onExit: () => {
        teardown();
      },
    };
  };
}

/**
 * Keyboard navigation for a suggestion list: wrap-around arrow keys, Enter to
 * pick, and a reset to the top whenever the query produces new items. Returns
 * the active index for the list to style; the popup drives it through `ref`.
 */
export function useSuggestionKeys<TItem>(
  items: TItem[],
  onSelect: (item: TItem) => void,
  ref: ForwardedRef<SuggestionListRef>,
): number {
  // The index is stored *with* the list it was chosen from, which makes "a new
  // list starts at the top" a derivation rather than a reset effect — the
  // suggestion plugin hands us a freshly built array whenever the query changes.
  const [picked, setPicked] = useState<{ items: TItem[]; index: number }>({ items, index: 0 });
  const index = picked.items === items ? picked.index : 0;

  useImperativeHandle(ref, () => ({
    onKeyDown: (e) => {
      if (!items.length) return false;
      if (e.key === "ArrowDown") {
        setPicked({ items, index: (index + 1) % items.length });
        return true;
      }
      if (e.key === "ArrowUp") {
        setPicked({ items, index: (index - 1 + items.length) % items.length });
        return true;
      }
      if (e.key === "Enter") {
        const it = items[index];
        if (it) onSelect(it);
        return true;
      }
      return false;
    },
  }));

  return index;
}
