import { useId, useState, type FormEvent } from "react";

import { ApiError, errorMessage, postJson } from "../api/client";
import type { Case, CaseCreate } from "../api/types";
import { setSelectedCase } from "../lib/ingest";

/**
 * Create the case the next ingest will file into.
 *
 * The basis is required and typed here rather than defaulted, because it is the
 * claim everything filed under this case rests on (spec 12) — a case created
 * with a placeholder is a case whose evidence nobody can say was lawful to
 * process. It is correctable later, and every correction is audited with the
 * old text (`CaseBasis`), which is what makes writing the real one now safe
 * rather than final.
 *
 * On success the new case becomes the selected one: creating a case is
 * something an operator does because they are about to add to it.
 */
export function NewCase({ onCreated }: { onCreated: () => void }) {
  const nameId = useId();
  const basisId = useId();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [basis, setBasis] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const body: CaseCreate = {
      name: name.trim(),
      authorization_basis: basis.trim(),
    };
    if (body.name === "" || body.authorization_basis === "") {
      setError("A case needs a name and the basis that permits processing its material.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const created = await postJson<Case>("/api/cases", body);
      setSelectedCase(created.id);
      setName("");
      setBasis("");
      setOpen(false);
      onCreated();
    } catch (failure) {
      setError(
        failure instanceof ApiError
          ? `Could not create the case (${String(failure.status)}): ${failure.detail}`
          : `Could not create the case: ${errorMessage(failure)}`,
      );
    } finally {
      setSaving(false);
    }
  }

  if (!open) {
    return (
      <button type="button" className="button-link" onClick={() => setOpen(true)}>
        New case
      </button>
    );
  }

  return (
    <form className="new-case" onSubmit={(event) => void submit(event)}>
      <label htmlFor={nameId}>Case name</label>
      <input
        id={nameId}
        type="text"
        maxLength={200}
        value={name}
        placeholder="What this case is called"
        onChange={(event) => setName(event.currentTarget.value)}
      />
      <label htmlFor={basisId}>Authorization basis</label>
      <textarea
        id={basisId}
        rows={3}
        maxLength={2000}
        value={basis}
        placeholder="What permits processing this material — warrant, consent, instruction — and its reference"
        onChange={(event) => setBasis(event.currentTarget.value)}
      />
      <p className="compact muted">
        Recorded with the case and carried by every audit entry under it. It can be corrected
        afterwards, and the correction keeps the previous text.
      </p>
      {error !== null && (
        <p className="notice error" role="alert">
          {error}
        </p>
      )}
      <div className="case-basis-actions">
        <button className="primary" type="submit" disabled={saving}>
          {saving ? "Creating…" : "Create case"}
        </button>
        <button
          type="button"
          disabled={saving}
          onClick={() => {
            setOpen(false);
            setError(null);
          }}
        >
          Cancel
        </button>
      </div>
    </form>
  );
}
