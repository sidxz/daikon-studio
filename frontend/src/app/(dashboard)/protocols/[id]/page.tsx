import { ProtocolDetail } from "@/features/protocols";
import { Suspense } from "react";

export default async function ProtocolDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  // useSearchParams (the ?tab= deep link) needs a suspense boundary in the app router.
  return (
    <Suspense fallback={null}>
      <ProtocolDetail protocolId={id} />
    </Suspense>
  );
}
