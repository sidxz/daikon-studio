"use client";

import { useThread } from "../hooks/use-studies";

/**
 * The study's answer to its own goal, once every step has run.
 *
 * Everything else on this page reports a step. The scientist asked a question in their own
 * words, and until this existed the answer was spread across five turns for them to assemble,
 * or buried in the thread summary, which is the agent's own running notes and reads like them.
 * The agent writes this on the turn that observes the last step and on no other, so it cannot
 * appear over a study that is still running.
 */
export function StudyConclusion({ studyId }: { studyId: string }) {
  const thread = useThread(studyId);
  // The latest, so a conclusion is rewritten rather than duplicated when further steps are
  // added to a finished study and it runs on.
  const conclusion = (thread.data?.turns ?? [])
    .map((turn) => turn.conclusion?.trim())
    .findLast((text): text is string => Boolean(text));
  if (!conclusion) return null;
  return (
    <section className="space-y-2">
      <h2 className="text-sm font-medium">What it found</h2>
      <p className="whitespace-pre-wrap text-sm">{conclusion}</p>
    </section>
  );
}
