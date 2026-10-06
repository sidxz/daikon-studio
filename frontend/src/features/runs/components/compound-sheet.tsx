import { StructureThumbnail } from "@/shared/components/chemistry/structure-thumbnail";
import { ReadoutValue } from "@/shared/components/readout-value";
import { Button } from "@/shared/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/shared/components/ui/sheet";
import type { ReadoutResponse } from "@/shared/lib/api/model";
import { showError } from "@/shared/lib/toast";
import { Check, Copy } from "lucide-react";
import { useState } from "react";
import { readoutDescription, resultTargets, uncertaintyInfo } from "../lib/result-presentation";
import type { TriageRow } from "../types";
import { ApplicabilityCell, ClassCell, ProbabilityCell } from "./result-cells";

export function CompoundSheet({
  row,
  readouts,
  engineId,
  selected,
  onSelect,
  onClose,
}: {
  row: TriageRow | null;
  readouts: ReadoutResponse[];
  engineId?: string;
  selected: boolean;
  onSelect: () => void;
  onClose: () => void;
}) {
  const [copiedStructure, setCopiedStructure] = useState<string | null>(null);
  const copied = copiedStructure === row?.structure;
  return (
    <Sheet
      open={row !== null}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <SheetContent className="w-full overflow-y-auto sm:max-w-md">
        <SheetHeader>
          <SheetTitle>
            {row?.compound_id ??
              (row?.input_row != null
                ? `Compound at input row ${row.input_row}`
                : "Compound details")}
          </SheetTitle>
          <SheetDescription>
            Model predictions for this compound. These values have not been measured.
          </SheetDescription>
        </SheetHeader>
        {row && (
          <div className="space-y-5 px-4 pb-6">
            <div className="flex justify-center rounded-lg border bg-muted/10 p-4">
              <StructureThumbnail smiles={row.structure} size={240} />
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-xs font-medium text-muted-foreground">SMILES</p>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={async () => {
                    try {
                      await navigator.clipboard.writeText(row.structure);
                      setCopiedStructure(row.structure);
                    } catch {
                      showError("Could not copy SMILES. Select and copy the text below.");
                    }
                  }}
                >
                  {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
                  {copied ? "Copied" : "Copy SMILES"}
                </Button>
              </div>
              <p className="select-text break-all rounded-md bg-muted/40 p-2 font-mono text-xs">
                {row.structure}
              </p>
            </div>
            {resultTargets(readouts).map((target) => (
              <section key={target.name} className="space-y-3 rounded-lg border p-3">
                <div>
                  <h3 className="break-words text-sm font-medium">{target.name}</h3>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {readoutDescription(target.readout)}
                  </p>
                </div>
                <dl className="space-y-3 text-sm">
                  {target.probability && (
                    <div className="grid grid-cols-2 items-center gap-3">
                      <dt className="text-muted-foreground">Probability</dt>
                      <dd>
                        <ProbabilityCell
                          value={row.readouts[target.probability.name]?.value ?? null}
                          cutoff={target.readout.threshold ?? 0.5}
                        />
                      </dd>
                    </div>
                  )}
                  <div className="grid grid-cols-2 items-center gap-3">
                    <dt className="text-muted-foreground">
                      {target.readout.type === "class" ? "Predicted class" : "Predicted value"}
                    </dt>
                    <dd>
                      {target.readout.type === "class" ? (
                        <ClassCell value={row.readouts[target.name]?.value ?? null} />
                      ) : (
                        <ReadoutValue
                          value={row.readouts[target.name]?.value}
                          unit={target.readout.unit}
                          precision={4}
                        />
                      )}
                    </dd>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <dt className="text-muted-foreground">Uncertainty</dt>
                    <dd>
                      {row.uncertainty?.[target.name] == null ? (
                        "Not reported"
                      ) : (
                        <ReadoutValue
                          value={row.uncertainty[target.name]}
                          unit={target.readout.type === "numeric" ? target.readout.unit : null}
                          precision={4}
                        />
                      )}
                    </dd>
                  </div>
                </dl>
                <p className="text-xs text-muted-foreground">
                  {uncertaintyInfo(engineId, target.readout).description}
                </p>
              </section>
            ))}
            <section className="space-y-2 rounded-lg border p-3">
              <h3 className="text-sm font-medium">Applicability</h3>
              <ApplicabilityCell value={row.applicability} />
              <p className="text-xs text-muted-foreground">
                Similarity to the nearest training compound. The applicability domain starts at 30%
                similarity.
              </p>
            </section>
            <Button
              variant={selected ? "outline" : "default"}
              className="w-full"
              onClick={onSelect}
            >
              {selected ? "Remove from selection" : "Select compound"}
            </Button>
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
}
