import { DatasetDetail } from "@/features/datasets";
import { Suspense } from "react";

export default async function DatasetDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  // useSearchParams (the ?tab= deep link) needs a suspense boundary in the app router.
  return (
    <Suspense fallback={null}>
      <DatasetDetail datasetId={id} />
    </Suspense>
  );
}
