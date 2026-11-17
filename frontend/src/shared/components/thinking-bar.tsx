/**
 * The one way the app says it is waiting on the model.
 *
 * A study's waits are long enough to watch, so they get the brand mark's
 * spectrum drifting through a still line rather than a spinner. The drawing is
 * two pseudo-elements in `globals.css` (`.thinking-bar`), the line and its
 * reflection, which is why there is no markup for them here.
 *
 * An `<output>`, which is a live region already, so the label is announced when
 * the wait begins; the bar itself is decoration and carries no text.
 */
export function ThinkingBar({ label }: { label: string }) {
  return (
    <output className="block space-y-2 py-1">
      <span className="thinking-bar block" aria-hidden="true" />
      <span className="block text-sm text-muted-foreground">{label}</span>
    </output>
  );
}
