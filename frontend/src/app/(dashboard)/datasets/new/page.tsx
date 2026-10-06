import { DatasetWizard } from "@/features/datasets";
import { Suspense } from "react";

export default function NewDatasetPage() {
  return (
    <Suspense fallback={null}>
      <DatasetWizard />
    </Suspense>
  );
}
