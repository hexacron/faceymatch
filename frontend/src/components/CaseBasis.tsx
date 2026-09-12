import { useState, type FormEvent } from "react";

import { ApiError, errorMessage, patchJson } from "../api/client";
import type { Case, CaseAmendment } from "../api/types";

/**
 * The authorization basis of the case an ingest is about to write into, and
 * the affordance to correct it.
 *
 * This sits next to every destination display rather than on a settings page
 * because it is the claim the evidence rests on (spec 12): the operator has to
 * be able to read it at the moment they are about to add to the case, and to
 * fix it there when it is wrong. Amending is an append to the record, not a
 * rewrite, which the copy says out loud — a basis nobody dares correct is
 * worse than one that carries its own history.
 *
 * Mount with `key={record.id}` so switching cases starts a fresh draft.
 */
export function CaseBasis({ record, onAmended }: { record: Case; onAmended: () => void }) {
  const [editing, setEditing] = useState(false);
  const [basis, setBasis] = useState(record.authorization_basis);
  const [reason, setReason] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [saving, setSaving] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const next = basis.trim();
    if (next === "") {
      // Refused here: an empty basis is not a correction, it is the removal of
      // the justification for everything already in the case.
      setError("An authorization basis is required: say what permits processing this case.");
      return;
    }
    const body: CaseAmendment = {
      authorization_basis: next,
      reason: reason.trim() === "" ? null : reason.trim(),
    };
    setSaving(true);
    setError(null);
    try {
      await patchJson<Case>(`/api/cases/${encodeURIComponent(record.id)}`, body);
      setEditing(false);
      setSaved(true);
      onAmended();
    } catch (failure) {
      if (failure instanceof ApiError && failure.status === 404) {
        setError("That case no longer exists, so there is nothing to amend. Refreshing the list.");
        onAmended();
      } else {
        setError(
          failure instanceof ApiError
            ? `Could not save the amendment (${String(failure.status)}): ${failure.detail}`
            : `Could not save the amendment: ${errorMessage(failure)}`,
        );
      }
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="case-basis">
      <div className="case-basis-head">
        <span className="case-basis-label" id={`basis-label-${record.id}`}>
          Authorization basis
        </span>
        {!editing && (
          <button
            type="button"
            className="button-link"
            onClick={() => {
              setBasis(record.authorization_basis);
              setReason("");
              setError(null);
              setSaved(false);
              setEditing(true);
            }}
          >
            Amend
          </button>
        )}
      </div>

      {editing ? (
        <form className="case-basis-form" onSubmit={(event) => void submit(event)}>
          <label>
            Corrected basis
            <textarea
              value={basis}
              rows={3}
              maxLength={2000}
              aria-describedby={`basis-audit-${record.id}`}
              onChange={(event) => setBasis(event.currentTarget.value)}
            />
          </label>
          <label>
            Reason for the correction <span className="muted">(optional)</span>
            <input
              type="text"
              maxLength={2000}
              value={reason}
              placeholder="Why the basis is being corrected"
              onChange={(event) => setReason(event.currentTarget.value)}
            />
          </label>
          <p className="compact muted" id={`basis-audit-${record.id}`}>
            The current text is kept: an amendment is appended to the audit log with both the old
            and the new basis, so nothing is overwritten and correcting a wrong basis is the safe
            move.
          </p>
          <div className="case-basis-actions">
            <button className="primary" type="submit" disabled={saving}>
              {saving ? "Saving amendment…" : "Save amendment"}
            </button>
            <button
              type="button"
              disabled={saving}
              onClick={() => {
                setEditing(false);
                setError(null);
              }}
            >
              Cancel
            </button>
          </div>
        </form>
      ) : (
        <p className="case-basis-text" aria-labelledby={`basis-label-${record.id}`}>
          {record.authorization_basis}
        </p>
      )}

      {error !== null && <p className="notice error" role="alert">{error}</p>}
      {saved && !editing && (
        <p className="compact muted" role="status">
          Amended. The previous basis stays in the audit log.
        </p>
      )}
    </div>
  );
}
