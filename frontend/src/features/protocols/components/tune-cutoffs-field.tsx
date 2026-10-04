"use client";

import { Checkbox } from "@/shared/components/ui/checkbox";
import { Label } from "@/shared/components/ui/label";

/**
 * The one training option that is not an engine setting: it changes how every
 * fit in the submission is scored, so the train and sweep forms share it.
 * Only a dataset with an active/inactive target has a cutoff to tune, so the
 * caller renders it only then.
 */
export function TuneCutoffsField({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-2">
        <Checkbox
          id="tune-cutoffs"
          checked={checked}
          onCheckedChange={(value) => onChange(value === true)}
        />
        <Label htmlFor="tune-cutoffs" className="font-normal">
          Tune decision cutoffs
        </Label>
      </div>
      <p className="text-xs text-muted-foreground">
        Chooses each active/inactive label's cutoff to maximize MCC on the validation set. Applies
        to the model and its baseline alike.
      </p>
    </div>
  );
}
