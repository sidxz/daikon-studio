import { type ExternalToast, toast } from "sonner";

/**
 * Thin wrappers over sonner so feature code never imports `toast` directly.
 * One place decides copy, duration and id conventions, and the toast library
 * can be swapped without a feature-wide edit.
 */

export type ToastId = string | number;

export function showSuccess(message: string, opts?: ExternalToast): ToastId {
  return toast.success(message, opts);
}

export function showError(message: string, opts?: ExternalToast): ToastId {
  return toast.error(message, opts);
}

export function showInfo(message: string, opts?: ExternalToast): ToastId {
  return toast.info(message, opts);
}

export function showWarning(message: string, opts?: ExternalToast): ToastId {
  return toast.warning(message, opts);
}

export function dismissToast(id?: ToastId): void {
  toast.dismiss(id);
}
