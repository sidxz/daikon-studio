import { StudyPage } from "@/features/studies";
import { Suspense } from "react";

export default async function StudyRoute({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return (
    <Suspense fallback={null}>
      <StudyPage studyId={id} />
    </Suspense>
  );
}
