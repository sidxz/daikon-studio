"use client";

import { QueryError } from "@/shared/components/query-error";
import { Button } from "@/shared/components/ui/button";
import { Skeleton } from "@/shared/components/ui/skeleton";
import { Textarea } from "@/shared/components/ui/textarea";
import type { SayRequest } from "@/shared/lib/api/model";
import { useQueryClient } from "@tanstack/react-query";
import { Loader2, X } from "lucide-react";
import { Fragment, useEffect, useRef, useState } from "react";
import { threadKey } from "../hooks/query-keys";
import { useSay, useStudy, useThread } from "../hooks/use-studies";
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
  // The questions still open: the newest agent turn that asked any, with nothing said since.
  const asked = turns.findLastIndex((t) => t.role === "agent" && t.questions.length > 0);
  const open =
    asked > turns.findLastIndex((t) => t.role === "scientist") ? turns[asked].questions : [];
  const typed = message.trim();

  const count = turns.length;
  useEffect(() => {
    // The newest turn is the one to read: keep it in view as the thread grows.
    if (count > 0 && list.current) list.current.scrollTop = list.current.scrollHeight;
  }, [count]);

  function send(body: SayRequest) {
    say.mutate(
      { ...body, about: about ?? undefined, provider: provider || undefined },
      {
        onSuccess: () => {
          setMessage("");
          setAnswers({});
        },
        onError: () => {
          // The scientist's words are kept before the agent is asked, so a turn the agent
          // failed may still have left them in the thread. The box keeps them too.
          void client.invalidateQueries({ queryKey: threadKey(studyId), exact: true });
        },
      },
    );
  }

  return (
    <section className="flex min-h-0 flex-1 flex-col gap-3">
      <h2 className="text-sm font-medium">Conversation</h2>
      {thread.data?.summary && (
        <div className="space-y-1 rounded-sm border border-dashed p-3">
          <h3 className="text-xs font-medium">Where this stands</h3>
          <p className="text-sm text-muted-foreground">{thread.data.summary}</p>
        </div>
      )}

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
              <TurnView studyId={studyId} turn={turn} />
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
                  <Button
                    size="sm"
                    disabled={say.isPending}
                    onClick={() =>
                      send({
                        answers: open.map((q) => ({
                          ask: q.ask,
                          answer: answers[q.ask] ?? q.default ?? "",
                        })),
                        message: typed || undefined,
                      })
                    }
                  >
                    Send answers
                  </Button>
                </div>
              )}
            </Fragment>
          ))
        )}
        {say.isPending && (
          <p className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" aria-hidden="true" />
            Daikon Studio is writing a reply
          </p>
        )}
      </div>

      <div className="space-y-2 border-t pt-3">
        {about && (
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <span className="min-w-0 rounded-sm border px-1.5 py-0.5">
              About {aboutStep(study?.steps ?? [], about)}
            </span>
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label="Clear the step"
              onClick={onClearAbout}
            >
              <X aria-hidden="true" />
            </Button>
          </div>
        )}
        <label className="sr-only" htmlFor={MESSAGE_ID}>
          Message to Daikon Studio
        </label>
        <Textarea
          id={MESSAGE_ID}
          rows={2}
          value={message}
          disabled={say.isPending}
          placeholder="Ask about the plan or a result, or say what to do next"
          // A message can hold a pasted sequence, one long word that would push the rail wide.
          className="wrap-anywhere"
          onChange={(event) => setMessage(event.target.value)}
        />
        {say.error && (
          <p className="text-sm text-destructive" role="alert">
            {say.error.message}
          </p>
        )}
        <div className="flex justify-end">
          <Button
            size="sm"
            disabled={say.isPending || !typed}
            onClick={() => send({ message: typed })}
          >
            Send
          </Button>
        </div>
      </div>
    </section>
  );
}
