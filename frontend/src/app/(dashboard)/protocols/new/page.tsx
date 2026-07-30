import { TrainProtocolForm } from "@/features/protocols";
import { Suspense } from "react";

export default function NewProtocolPage() {
  // useSearchParams needs a suspense boundary in the app router.
  return (
    <Suspense fallback={null}>
      <TrainProtocolForm />
    </Suspense>
  );
}
