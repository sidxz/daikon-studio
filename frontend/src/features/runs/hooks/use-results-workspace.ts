"use client";

import { useEffect, useRef, useState } from "react";

/** Keep the same grid mounted as the run header scrolls away or the table expands. */
export function useResultsWorkspace() {
  const workspaceRef = useRef<HTMLDivElement>(null);
  const [maximized, setMaximized] = useState(false);
  const [height, setHeight] = useState<number>();

  useEffect(() => {
    const workspace = workspaceRef.current;
    if (!workspace) return;
    let frame = 0;
    const measure = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        // An inactive tab is still mounted to preserve its grid state.
        if (!workspace.getClientRects().length) return;
        const bounds = workspace.getBoundingClientRect();
        setHeight(
          Math.max(320, Math.min(window.innerHeight - 8, bounds.bottom) - Math.max(8, bounds.top)),
        );
      });
    };
    const observer = new ResizeObserver(measure);
    observer.observe(workspace);
    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, true);
    measure();

    const scrollToTable = (event: WheelEvent) => {
      if (
        maximized ||
        event.ctrlKey ||
        event.deltaY <= 0 ||
        Math.abs(event.deltaX) > Math.abs(event.deltaY)
      )
        return;
      if (!(event.target instanceof Element) || !event.target.closest(".ag-body-viewport")) return;
      const remaining = workspace.getBoundingClientRect().top - 8;
      if (remaining <= 1) return;
      // Give the page the first part of a downward scroll, then let AG Grid
      // handle its rows once the results toolbar reaches the top.
      event.preventDefault();
      const delta =
        event.deltaY *
        (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? window.innerHeight : 1);
      let parent = workspace.parentElement;
      while (parent) {
        if (
          parent.scrollHeight > parent.clientHeight &&
          /auto|scroll/.test(getComputedStyle(parent).overflowY)
        ) {
          parent.scrollBy({ top: Math.min(delta, remaining), behavior: "instant" });
          return;
        }
        parent = parent.parentElement;
      }
      window.scrollBy({ top: Math.min(delta, remaining), behavior: "instant" });
    };
    workspace.addEventListener("wheel", scrollToTable, { passive: false, capture: true });
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure, true);
      workspace.removeEventListener("wheel", scrollToTable, true);
    };
  }, [maximized]);

  useEffect(() => {
    if (!maximized) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    // Keep keyboard focus in the visible workspace. Dialogs and menus opened
    // from the grid can still use their portals outside this ancestor chain.
    const background: { element: HTMLElement; inert: boolean }[] = [];
    let branch: HTMLElement | null = workspaceRef.current;
    while (branch && branch !== document.body) {
      for (const sibling of branch.parentElement?.children ?? []) {
        if (sibling instanceof HTMLElement && sibling !== branch) {
          background.push({ element: sibling, inert: sibling.inert });
          sibling.inert = true;
        }
      }
      branch = branch.parentElement;
    }
    const exit = (event: KeyboardEvent) => {
      // A popup or compound inspector gets the first Escape.
      if (
        event.defaultPrevented ||
        (event.target instanceof Element &&
          event.target.closest('[role="dialog"], [role="menu"], .ag-popup'))
      )
        return;
      if (event.key === "Escape") setMaximized(false);
    };
    document.addEventListener("keydown", exit);
    return () => {
      document.body.style.overflow = previousOverflow;
      for (const { element, inert } of background) element.inert = inert;
      document.removeEventListener("keydown", exit);
    };
  }, [maximized]);

  return { workspaceRef, height, maximized, setMaximized };
}
