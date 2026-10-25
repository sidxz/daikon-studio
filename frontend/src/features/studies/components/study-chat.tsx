"use client";

import { QueryError } from "@/shared/components/query-error";
import { ThinkingBar } from "@/shared/components/thinking-bar";
import { Button } from "@/shared/components/ui/button";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { Textarea } from "@/shared/components/ui/textarea";
import type { SayRequest } from "@/shared/lib/api/model";
import { useQueryClient } from "@tanstack/react-query";
import { SendHorizontal, X } from "lucide-react";
import { Fragment, useEffect, useRef, useState } from "react";
import { threadKey } from "../hooks/query-keys";
import { useSay, useStudy, useThread } from "../hooks/use-studies";
import { RUN_UNDER_WAY, STEPS_REASON_ID } from "../types";
import { Fold } from "./fold";
import { QuestionCard } from "./question-card";
import { TurnView, aboutStep } from "./turn-view";

/** The composer's id, so "Talk about this result" on a card can put the cursor in it. */
export const MESSAGE_ID = "study-message";

/**
 * The study's one conversation, from the agent's first questions to its last observation, and
 * the one place to talk to it. Docked beside the steps, because a claim is only checkable
 * while the result it points at is still on screen.
 */
export function StudyChat({
  studyId,
  provider,
  about,
  onClearAbout,
}: {
  studyId: string;
  /** The model the scientist chose, sent with every turn; empty for the server's default. */
  provider: string;
  /** The step the next message is about, set from its card; null for the whole study. */
  about: string | null;
  onClearAbout: () => void;
}) {
  const thread = useThread(studyId);
  const { data: study } = useStudy(studyId);
  const say = useSay(studyId);
  const client = useQueryClient();
  const [message, setMessage] = useState("");
  // Keyed by the question's words, which is what travels back to the agent.
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const list = useRef<HTMLDivElement>(null);

  const turns = thread.data?.turns ?? [];
  // The questions still open: the newest agent turn that asked any, with nothing said since,
  // while the plan is still a draft. Starting it settles whatever was left unanswered.
  const asked = turns.findLastIndex((t) => t.role === "agent" && t.questions.length > 0);
  const open =
    study?.status === "draft" && asked > turns.findLastIndex((t) => t.role === "scientist")
      ? turns[asked].questions
      : [];
  const replies = open.map((q) => ({ ask: q.ask, answer: answers[q.ask] ?? q.default ?? "" }));
  const typed = message.trim();
  const running = study?.running ?? false;

  // A run writes its turns while no request of ours is open, and every write moves the study's
  // `updated_at`: read the thread again when it moves. Not on the first sight of the study,
  // which the thread's own first read covers.
  const updated = study?.updated_at;
  const seen = useRef(updated);
  useEffect(() => {
    if (updated && seen.current && updated !== seen.current) {
      void client.invalidateQueries({ queryKey: threadKey(studyId), exact: true });
    }
    seen.current = updated;
  }, [client, studyId, updated]);

  const count = turns.length;
  // The bar is the last thing in the list, so a wait that begins moves the bottom just as a
  // new turn does. Both scroll, or answering a question leaves the bar below the fold.
  const waiting = say.isPending || running;
  useEffect(() => {
    // The newest turn, or the wait for the next one, is what to keep in view as the list grows.
    if ((count > 0 || waiting) && list.current) list.current.scrollTop = list.current.scrollHeight;
  }, [count, waiting]);

  /**
   * The one way a turn is sent. The open questions' answers go with every message, so a choice
   * made on a card is never lost to pressing Send rather than Send answers.
   */
  function send(text: string | undefined) {
    const answering = replies.length > 0;
    const body: SayRequest = {
      message: text,
      answers: answering ? replies : undefined,
      about: about ?? undefined,
      provider: provider || undefined,
    };
    say.mutate(body, {
      onSuccess: () => {
        setMessage("");
        if (answering) setAnswers({});
      },
      onError: () => {
        // The scientist's words are kept before the agent is asked, so a turn the agent
        // failed may still have left them in the thread. The box keeps them too.
        void client.invalidateQueries({ queryKey: threadKey(studyId), exact: true });
      },
    });
  }

  return (
    <section className="flex min-h-0 flex-1 flex-col gap-3">
      <h2 className="text-sm font-medium">Conversation</h2>

      <div ref={list} className="min-h-0 flex-1 space-y-2 overflow-y-auto">
        {thread.isLoadingError ? (
          <QueryError
            title="Could not load the conversation"
            retry={thread.refetch}
            retrying={thread.isRefetching}
          />
        ) : thread.isLoading ? (
          <Skeleton className="h-24 w-full rounded-sm" />
        ) : turns.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Daikon Studio has not replied yet. Send a message to ask again.
          </p>
        ) : (
          turns.map((turn, index) => (
            <Fragment key={`${turn.at}-${index}`}>
              <TurnView
                studyId={studyId}
                turn={turn}
                answering={index === asked && open.length > 0}
              />
              {index === asked && open.length > 0 && (
                <div className="space-y-2">
                  <ul className="space-y-2">
                    {open.map((question) => (
                      <QuestionCard
                        key={question.ask}
                        question={question}
                        answer={answers[question.ask] ?? question.default ?? ""}
                        disabled={say.isPending}
                        onAnswer={(value) =>
                          setAnswers((previous) => ({ ...previous, [question.ask]: value }))
                        }
                      />
                    ))}
                  </ul>
                  {/* Never shown during a run: a run needs a started study, and starting it
                      closes the questions. */}
                  <Button
                    size="sm"
                    disabled={say.isPending}
                    onClick={() => send(typed || undefined)}
                  >
                    Send answers
                  </Button>
                </div>
              )}
            </Fragment>
          ))
        )}
        {/* One bar for both waits: a step running is the model thinking, and a turn in
            flight is the model writing. Only one can be open at a time. */}
        {waiting && (
          <ThinkingBar
            label={say.isPending ? "Daikon Studio is writing a reply" : "Daikon Studio is thinking"}
          />
        )}
      </div>

      {/* Folded, above the composer: it can run to a page. Open, it scrolls in its own box,
          so the thread keeps its room. */}
      {thread.data?.summary && (
        <Fold label="Where this stands">
          {/* Its own scroller: this one sits outside the thread's, and a long summary would
              otherwise push the composer off the rail. */}
          <p className="max-h-40 overflow-y-auto">{thread.data.summary}</p>
        </Fold>
      )}

      <div className="space-y-2 border-t pt-3">
        {about && (
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <span className="min-w-0 rounded-sm border px-1.5 py-0.5">
              About {aboutStep(study?.steps ?? [], about)}
            </span>
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label="Stop sending this with the step"
              onClick={onClearAbout}
            >
              <X aria-hidden="true" />
            </Button>
          </div>
        )}
        <label className="sr-only" htmlFor={MESSAGE_ID}>
          Message to Daikon Studio
        </label>
        {/* Send beside the box, not under it: a row of its own cost the thread 40px of the
            rail, and the control is the same one icon every message. */}
        <div className="flex items-end gap-2">
          <Textarea
            id={MESSAGE_ID}
            rows={2}
            value={message}
            disabled={say.isPending}
            placeholder="Ask about the plan or a result, or say what to do next"
            // A message can hold a pasted sequence, one long word that would push the rail wide.
            className="min-w-0 flex-1 wrap-anywhere"
            onChange={(event) => setMessage(event.target.value)}
          />
          <Button
            size="icon-sm"
            aria-label="Send"
            title={running ? RUN_UNDER_WAY : "Send"}
            disabled={say.isPending || !typed || running}
            aria-describedby={running ? STEPS_REASON_ID : undefined}
            onClick={() => send(typed)}
          >
            <SendHorizontal className="size-4" aria-hidden="true" />
          </Button>
        </div>
        {say.error && (
          <p className="text-sm text-destructive" role="alert">
            {say.error.message}
          </p>
        )}
      </div>
    </section>
  );
}
