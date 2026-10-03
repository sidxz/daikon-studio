import { buttonVariants } from "@/shared/components/ui/button";
import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex min-h-screen items-center justify-center p-6">
      <div className="max-w-md space-y-3 text-center">
        <p className="text-lg font-semibold">This page does not exist</p>
        <Link href="/" className={buttonVariants({ variant: "outline", size: "sm" })}>
          Go to the home page
        </Link>
      </div>
    </div>
  );
}
