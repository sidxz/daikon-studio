import { PlanComposer } from "@/features/studies";
import { Suspense } from "react";

export default async function StudyPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return (
    <Suspense fallback={null}>
      <PlanComposer studyId={id} />
    </Suspense>
  );
}
