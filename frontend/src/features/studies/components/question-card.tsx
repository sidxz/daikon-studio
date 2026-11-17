"use client";

import { Button } from "@/shared/components/ui/button";
import { Input } from "@/shared/components/ui/input";
import { Check } from "lucide-react";
import { useId } from "react";
import type { Question } from "../types";
import { NOT_SURE } from "../types";

/**
 * One question, with its shortcuts and a field.
 *
 * The options fill the field rather than living in separate state, so what will
 * be sent is exactly what is on screen and a chemist can take a shortcut and
 * then qualify it. One answer per question, visible, editable.
 *
 * "Not sure" is added here and never comes from the model, because it is the
 * answer most likely to be true and least likely to be offered. Whether the
 * assay changed between two runs three months apart is often genuinely unknown,
 * and the plan is supposed to settle that with a step or name it as an
 * assumption rather than have someone guess into a yes.
 *
 * A refinement arrives with the planner's own answer. That answer is frequently
 * not one of the options but a blend of two, qualified -- "the 50 and 10 floors
 * as hard flags, plus a fold-change flag for anything that halves either value".
 * Rather than force it to be a single option and lose the qualification, the
 * assumption is put into the row as its own first choice. It therefore always
 * highlights, and it stays one click away after you have changed your mind.
 */
export function QuestionCard({
  question,
  answer,
  onAnswer,
  disabled,
}: {
  question: Question;
  answer: string;
  onAnswer: (value: string) => void;
  disabled?: boolean;
}) {
  const fieldId = useId();
  const assumed = question.default ?? null;
  const offered = question.options ?? [];
  const options = [
    ...(assumed && !offered.includes(assumed) ? [assumed] : []),
    ...offered,
    NOT_SURE,
  ];
  const untouched = assumed !== null && answer.trim() === assumed;

  return (
    <li className="space-y-2 rounded-sm border p-3">
      <div>
        <p className="text-sm font-medium">{question.ask}</p>
        <p className="mt-0.5 text-xs text-muted-foreground">{question.why}</p>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {options.map((option) => {
          const chosen = answer.trim() === option;
          const isAssumption = option === assumed;
          return (
            <Button
              key={option}
              type="button"
              size="sm"
              variant={chosen ? "default" : "outline"}
              aria-pressed={chosen}
              disabled={disabled}
              // The shared button is `whitespace-nowrap shrink-0`, which is right
              // for a label someone wrote and wrong for one a model did: these
              // run to a full sentence, and a single option wider than the row
              // took the whole page sideways. Let them wrap and cap them instead.
              className="h-auto max-w-full py-1 text-left leading-snug whitespace-normal"
              onClick={() => onAnswer(chosen ? "" : option)}
            >
              {isAssumption && chosen && <Check className="size-3.5 shrink-0" aria-hidden="true" />}
              {option}
            </Button>
          );
        })}
      </div>

      <Input
        id={fieldId}
        value={answer}
        disabled={disabled}
        placeholder="Or answer in your own words"
        aria-label={`Your answer to: ${question.ask}`}
        onChange={(event) => onAnswer(event.target.value)}
      />

      {assumed !== null && (
        <p className="text-xs text-muted-foreground" aria-live="polite">
          {untouched
            ? "Default assumption used in the analysis plan unless revised."
            : "Default assumption revised."}
        </p>
      )}
    </li>
  );
}
