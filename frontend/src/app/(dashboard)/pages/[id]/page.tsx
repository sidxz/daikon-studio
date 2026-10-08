import { PageContainer } from "@/features/pages";
import { Suspense } from "react";

export default async function NotebookPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  // PageContainer reads its one-shot ?edit=1 handoff through useSearchParams.
  return (
    <Suspense fallback={null}>
      <PageContainer id={id} />
    </Suspense>
  );
}
