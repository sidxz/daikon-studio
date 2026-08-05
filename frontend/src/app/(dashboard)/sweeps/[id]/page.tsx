import { SweepDetail } from "@/features/sweeps";
export default async function SweepDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <SweepDetail id={id} />;
}
