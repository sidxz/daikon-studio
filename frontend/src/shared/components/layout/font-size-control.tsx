"use client";

import { Slider } from "@/shared/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/shared/components/ui/tooltip";
import {
  FONT_SCALE_DEFAULT,
  FONT_SCALE_MAX,
  FONT_SCALE_MIN,
  FONT_SCALE_STEP,
  useFontScaleStore,
} from "@/shared/lib/stores/font-scale-store";
import { RotateCcw } from "lucide-react";

/**
 * Root font-size, as a percentage of the browser default; every rem-based
 * utility scales off it. Font *family* is not here -- it lives on /settings,
 * because it is a preference you set once, not one you reach for.
 *
 * Wraps its own TooltipProvider: the shared Tooltip has none embedded.
 */
export function FontSizeControl() {
  const scale = useFontScaleStore((s) => s.scale);
  const setScale = useFontScaleStore((s) => s.setScale);
  const reset = useFontScaleStore((s) => s.reset);
  const isDefault = scale === FONT_SCALE_DEFAULT;

  return (
    <TooltipProvider delayDuration={300}>
      <div className="flex items-center gap-2 px-1">
        <span aria-hidden className="text-xs font-semibold text-muted-foreground">
          A
        </span>
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center">
              <Slider
                value={[scale]}
                min={FONT_SCALE_MIN}
                max={FONT_SCALE_MAX}
                step={FONT_SCALE_STEP}
                onValueChange={([value]) => setScale(value)}
                onDoubleClick={reset}
                aria-label="Text size"
                className="w-24"
              />
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            Text size {scale}%{isDefault ? "" : " · double-click to reset"}
          </TooltipContent>
        </Tooltip>
        <span aria-hidden className="text-base font-semibold text-muted-foreground">
          A
        </span>
        {!isDefault && (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={reset}
                aria-label="Reset text size to 100%"
                className="rounded p-0.5 text-muted-foreground transition-colors hover:text-foreground"
              >
                <RotateCcw className="size-3.5" />
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom">Reset to 100%</TooltipContent>
          </Tooltip>
        )}
      </div>
    </TooltipProvider>
  );
}
