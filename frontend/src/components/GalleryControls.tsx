import { useId, useState, type FormEvent } from "react";

import { ApiError, deleteJson, errorMessage, patchJson, postJson } from "../api/client";
import type {
  Person,
  PersonPurgeResult,
  PersonUpdate,
  Template,
  TemplateRevoke,
} from "../api/types";
import { truncateHash } from "../lib/display";

/**
 * The three ways an operator takes a face, or a person, out of the gallery.
 *
 * They are different acts and the copy has to keep them apart. Revoking is
 * about one template: that embedding stops being compared, which is the only
 * fix for a duplicate face held by two persons — while both hold it, every
 * match against it ties and the margin rule refuses to name either. Parking a
 * person (`do_not_enroll`) is about the person: no new template may be created
 * for them and they drop out of matching, but the templates they already have
 * stay exactly where they are. Deleting a person is the only one that destroys
 * anything: the person row and every claim naming them go, in every case.
 *
 * The first two hide nothing. A revoked template keeps its row, its crop and
 * its place in the audit log; the log is append-only (invariant 6), so there is
 * no undo to offer and the confirm step says so instead of implying one. The
 * delete has no undo either, and one click stands between the operator and it.
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

/**
 * What parking did, said in terms of what it did not do. The template clause
 * is conditional because "the 0 templates they already have" is noise: with
 * nothing enrolled, the distinction worth drawing is the other one.
 */
function parkedText(person: Person): string {
  const opening = `${person.display_name} is parked: no new template can be created for them and they are held out of matching.`;
  if (person.template_count === 0) {
    return `${opening} They hold no template, so nothing was taken away — parking withholds the person, revoking is what removes a face.`;
  }
  const one = person.template_count === 1;
  return `${opening} The ${String(person.template_count)} template${one ? "" : "s"} they already have ${one ? "stays" : "stay"} in place — revoke a template when the face itself has to leave the gallery.`;
}

/**
 * Park or unpark a person.
 *
 * A checkbox rather than a confirm step: unlike a revoke this is reversible in
 * one click, and the description carries the distinction that matters — it
 * withholds the person from matching and from future enrolment without taking
 * a single template away.
 */
export function DoNotEnrollToggle({
  person,
  onOutcome,
  onRefresh,
}: {
  person: Person;
  onOutcome: (notice: GalleryNotice) => void;
  onRefresh: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const helpId = useId();

  async function apply(next: boolean): Promise<void> {
    const body: PersonUpdate = { do_not_enroll: next };
    setBusy(true);
    try {
      await patchJson<Person>(`/api/persons/${encodeURIComponent(person.id)}`, body);
      onOutcome({
        tone: next ? "attention" : "success",
        text: next ? parkedText(person) : `${person.display_name} can be enrolled again, and their active templates are back in the matching gallery.`,
      });
      onRefresh();
    } catch (failure) {
      if (failure instanceof ApiError && failure.status === 404) {
        onOutcome({
          tone: "attention",
          text: "That person no longer exists, so there is nothing to change. Reloaded from the server.",
        });
        onRefresh();
      } else {
        onOutcome(failureNotice(`change enrollment for ${person.display_name}`, failure));
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="person-control">
      <label className="checkbox-label">
        <input
          type="checkbox"
          checked={person.do_not_enroll}
          disabled={busy}
          aria-describedby={helpId}
          onChange={(event) => void apply(event.currentTarget.checked)}
        />
        <span className="checkbox-text">Do not enroll — park this person</span>
      </label>
      <p className="compact muted" id={helpId}>
        Parking blocks new templates for this person and holds them out of matching. It removes
        nothing: the templates they already have stay in place, so revoke a template when the face
        itself has to leave the gallery.
      </p>
    </div>
  );
}

/**
 * Delete a person from the library (spec 12).
 *
 * A bin icon that turns into "Delete / Cancel" for one click. No typed name,
 * no written justification: this is the operator clearing their own gallery,
 * and a form that has to be argued with gets clicked through rather than read.
 * The audit entry is the record that it happened.
 *
 * It is still the only destructive act in the UI, so it says what it removes
 * on the confirm step and the caller reloads the list afterwards rather than
 * guessing at the new one.
 */
export function DeletePersonButton({
  person,
  onOutcome,
  onDeleted,
}: {
  person: Person;
  onOutcome: (notice: GalleryNotice) => void;
  onDeleted: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);

  async function remove(): Promise<void> {
    setBusy(true);
    try {
      const result = await deleteJson<PersonPurgeResult>(
        `/api/persons/${encodeURIComponent(person.id)}`,
      );
      onOutcome({
        tone: "success",
        text: `${person.display_name} is deleted: ${String(result.templates)} template(s) and every identity, decision and match naming them went too. The media, detections and crops are kept.`,
      });
      onDeleted();
    } catch (failure) {
      if (failure instanceof ApiError && failure.status === 404) {
        // The intended state is the actual state; another window got there first.
        onDeleted();
      } else {
        onOutcome(failureNotice(`delete ${person.display_name}`, failure));
      }
    } finally {
      setBusy(false);
      setConfirming(false);
    }
  }

  if (!confirming) {
    return (
      <button
        type="button"
        className="icon-button danger"
        title={`Delete ${person.display_name}`}
        aria-label={`Delete ${person.display_name}`}
        onClick={() => setConfirming(true)}
      >
        <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
          <path
            d="M6 2h4v1h4v1.5H2V3h4V2zM3.5 6h9l-.7 8H4.2L3.5 6zM6.5 7.5v5M9.5 7.5v5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.2"
          />
        </svg>
      </button>
    );
  }

  return (
    <span className="delete-confirm">
      <button type="button" className="danger" disabled={busy} onClick={() => void remove()}>
        {busy ? "Deleting…" : "Delete"}
      </button>
      <button type="button" disabled={busy} onClick={() => setConfirming(false)}>
        Cancel
      </button>
    </span>
  );
}
