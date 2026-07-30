import { ProtocolDetail } from "@/features/protocols";

export default async function ProtocolDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <ProtocolDetail protocolId={id} />;
}
