import { DatasetDetail } from "@/features/datasets";

export default async function DatasetDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <DatasetDetail datasetId={id} />;
}
