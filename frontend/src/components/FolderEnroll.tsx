import { useState } from "react";

import { ApiError, errorMessage } from "../api/client";
import type { FolderEnroll as FolderEnrollResult, MediaImport } from "../api/types";
import { enrollFolder, importFolder } from "../lib/ingest";

/** Skips beyond this are counted, not listed: the panel is a report, not a log. */
const SKIPS_SHOWN = 20;

/**
 * A curated `Person Name/*.jpg` tree, in two ordered operator actions.
 *
 * Two and not one because a template can only be built from a stored detection:
 * the import registers and hashes the files, the worker processes them, and only
 * then is there a face to enrol. Running step 2 early is allowed and honest —
 * every file comes back skipped as "still processing".
 *
 * The path is typed rather than picked because a browser cannot hand a directory
 * path to the backend; the backend reads it from this machine's filesystem.
 */
export function FolderEnroll({ caseId, onImported }: { caseId: string; onImported: () => void }) {
  const [path, setPath] = useState("");
  const [reason, setReason] = useState("");
  // Ticked by default: a panel titled "Folder of faces" is opened by an operator who wants
  // faces. The API default stays false, so nothing else that calls the endpoint destroys
  // anything it was not asked to.
  const [facesOnly, setFacesOnly] = useState(true);
  const [busy, setBusy] = useState<"import" | "enroll" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [imported, setImported] = useState<MediaImport | null>(null);
  const [enrolled, setEnrolled] = useState<FolderEnrollResult | null>(null);

  const noCase = caseId === "";
  const noPath = path.trim() === "";
  const reasonMissing = reason.trim() === "";

  function report(action: string, failure: unknown): void {
    setError(
      failure instanceof ApiError
        ? `Could not ${action} (${String(failure.status)}): ${failure.detail}`
        : `Could not ${action}: ${errorMessage(failure)}`,
    );
  }

  async function runImport(): Promise<void> {
    setBusy("import");
    setError(null);
    try {
      setImported(await importFolder(caseId, path.trim(), facesOnly));
      onImported();
    } catch (failure) {
      report("import that folder", failure);
    } finally {
      setBusy(null);
    }
  }

  async function runEnroll(): Promise<void> {
    setBusy("enroll");
    setError(null);
    try {
      setEnrolled(await enrollFolder(caseId, path.trim(), reason.trim()));
    } catch (failure) {
      report("enrol from that folder", failure);
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="panel">
      <h3>Folder of faces</h3>
      <p className="compact muted">
        The backend reads this path from this machine, so type it rather than picking it: a browser
        file picker cannot hand over a directory path. Import registers and hashes every image;
        enrolling then reads the faces already stored for them and creates one person per immediate
        subfolder, named by that folder.
      </p>

      {noCase && <p className="notice attention">Select a case before importing evidence.</p>}

      <div className="form-grid">
        <label>
          Folder path
          <input
            type="text"
            placeholder="/Users/you/faces"
            value={path}
            onChange={(event) => setPath(event.currentTarget.value)}
          />
        </label>
        <label>
          Reason <span className="muted">(required to enrol)</span>
          <textarea
            rows={2}
            maxLength={2000}
            value={reason}
            placeholder="Why this folder is being treated as identified faces — this is what the audit log records"
            onChange={(event) => setReason(event.currentTarget.value)}
          />
        </label>
        <button
          className="primary"
          type="button"
          disabled={busy !== null || noCase || noPath}
          onClick={() => void runImport()}
        >
          {busy === "import" ? "Importing…" : "Import folder"}
        </button>
        <button
          className="primary"
          type="button"
          disabled={busy !== null || noCase || noPath || reasonMissing}
          onClick={() => void runEnroll()}
        >
          {busy === "enroll" ? "Enrolling…" : "Enrol from folder names"}
        </button>
      </div>

      <label className="checkbox-label">
        <input
          type="checkbox"
          checked={facesOnly}
          onChange={(event) => setFacesOnly(event.currentTarget.checked)}
        />
        Only keep files with a face
      </label>
      <p className="compact muted">
        Every file is registered and hashed first, because that is the only way to find out what
        is in it. A file the detector finds no face in is then deleted — its row, its bytes and
        its derived rows — and the deletion is recorded in the audit log.
      </p>

      {reasonMissing && (
        <p className="field-error">
          Bulk enrolment is only accepted with a stated reason: it creates gallery templates.
        </p>
      )}

      {imported !== null && (
        <p className="compact" role="status">
          {imported.media_ids.length} files registered, {imported.job_ids.length} queued,{" "}
          {imported.reused} already in this case.
          {facesOnly ? " Files with no face are removed as they are processed." : ""}
        </p>
      )}

      {enrolled !== null && (
        <div role="status">
          <p className="compact">
            {enrolled.templates_created} templates across {enrolled.persons.length} people from{" "}
            {enrolled.files_seen} files.
          </p>
          <ul className="compact">
            {enrolled.persons.map((person) => (
              <li key={person.person_id}>
                {person.display_name} — {person.templates_created} templates
                {person.created ? " (new)" : ""}
              </li>
            ))}
          </ul>
          {enrolled.skipped.length > 0 && (
            <ul className="compact muted">
              {enrolled.skipped.slice(0, SKIPS_SHOWN).map((skip) => (
                <li key={skip.file}>
                  {skip.file} — {skip.reason}
                </li>
              ))}
              {enrolled.skipped.length > SKIPS_SHOWN && (
                <li>+{enrolled.skipped.length - SKIPS_SHOWN} more</li>
              )}
            </ul>
          )}
        </div>
      )}

      {error !== null && (
        <p className="notice error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
