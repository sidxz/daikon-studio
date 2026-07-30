import { PredictWizard } from "@/features/runs";
import { Suspense } from "react";
export default function NewRunPage() {
  return (
    <Suspense fallback={null}>
      <PredictWizard />
    </Suspense>
  );
}
