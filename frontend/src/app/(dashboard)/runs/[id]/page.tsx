import { RunDetail } from "@/features/runs";
import { Suspense } from "react";
export default async function RunDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  // useSearchParams needs a suspense boundary in the app router.
  return (
    <Suspense fallback={null}>
      <RunDetail runId={id} />
    </Suspense>
  );
}
