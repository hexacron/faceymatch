import { useId, useState, type FormEvent } from "react";

import { ApiError, errorMessage, postJson } from "../api/client";
import type { Template, TemplateRevoke } from "../api/types";
import { truncateHash } from "../lib/display";

/**
 * Taking a face out of the matching gallery, from the page that lists it.
 *
 * Revoking is about one template: that embedding stops being compared, which
 * is the only fix for a duplicate face held by two persons — while both hold
 * it, every match against it ties and the spec 6.4 margin rule refuses to name
 * either. It hides nothing: the row, the crop and the audit entry all stay, and
 * because the log is append-only (invariant 6) there is no undo to offer, so
 * the confirm step says so rather than implying one.
 */

/** An outcome the operator needs to read, rendered by the page that owns it. */
export type GalleryNotice = {
  tone: "success" | "attention" | "error";
  text: string;
};

function failureNotice(action: string, failure: unknown): GalleryNotice {
  return {
    tone: "error",
    text:
      failure instanceof ApiError
        ? `Could not ${action} (${String(failure.status)}): ${failure.detail}`
        : `Could not ${action}: ${errorMessage(failure)}`,
  };
}

/**
 * Revoke one active template, behind a confirm step that states the
 * consequence.
 *
 * Render it only for an active template: a revoked one has nothing left to
 * confirm. Every path that changed the server — including the 409 where
 * someone else revoked first — calls `onRefresh`, because the person's
 * enrollment state is computed from active templates and guessing at it
 * locally is how a page comes to claim a gallery membership the matcher does
 * not agree with.
 */
export function RevokeTemplateControl({
  personId,
  template,
  onOutcome,
  onRefresh,
}: {
  personId: string;
  template: Template;
  onOutcome: (notice: GalleryNotice) => void;
  onRefresh: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const consequenceId = useId();
  const shortId = truncateHash(template.id);

  async function revoke(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const body: TemplateRevoke = { reason: reason.trim() === "" ? null : reason.trim() };
    setBusy(true);
    try {
      await postJson<Template>(
        `/api/persons/${encodeURIComponent(personId)}/templates/${encodeURIComponent(template.id)}/revoke`,
        body,
      );
      onOutcome({
        tone: "success",
        text: `Template ${shortId} is revoked: that face has left the matching gallery. Its row, its crop and the audit trail are kept.`,
      });
      onRefresh();
    } catch (failure) {
      if (failure instanceof ApiError && failure.status === 409) {
        // Not a failure: the intended state is the actual state, someone else
        // just got there first.
        onOutcome({
          tone: "attention",
          text: `Template ${shortId} was already revoked — another session got there first, so there is nothing left to do. Reloaded from the server.`,
        });
        onRefresh();
      } else if (failure instanceof ApiError && failure.status === 404) {
        onOutcome({
          tone: "attention",
          text: `Template ${shortId} is no longer one of this person's templates, so it cannot be revoked here. Reloaded from the server.`,
        });
        onRefresh();
      } else {
        onOutcome(failureNotice(`revoke template ${shortId}`, failure));
      }
    } finally {
      setBusy(false);
    }
  }

  if (!confirming) {
    return (
      <div className="template-actions">
        <button
          type="button"
          className="danger"
          aria-label={`Revoke template ${shortId}`}
          onClick={() => {
            setReason("");
            setConfirming(true);
          }}
        >
          Revoke…
        </button>
      </div>
    );
  }

  return (
    <form
      className="template-actions revoke-form"
      aria-label={`Confirm revoking template ${shortId}`}
      onSubmit={(event) => void revoke(event)}
    >
      <p className="compact" id={consequenceId}>
        This face leaves the matching gallery: nothing will ever be compared against it again.
        The template row, its crop and the audit trail are all kept. Revoking is one-way — there
        is no undo, so bringing this face back means enrolling it again from a crop.
      </p>
      <label>
        Reason <span className="muted">(optional)</span>
        <input
          type="text"
          maxLength={2000}
          value={reason}
          placeholder="Why this face is leaving the gallery"
          aria-describedby={consequenceId}
          onChange={(event) => setReason(event.currentTarget.value)}
        />
      </label>
      <div className="template-action-row">
        <button type="submit" className="danger" disabled={busy}>
          {busy ? "Revoking…" : "Revoke this template"}
        </button>
        <button type="button" disabled={busy} onClick={() => setConfirming(false)}>
          Cancel
        </button>
      </div>
    </form>
  );
}
