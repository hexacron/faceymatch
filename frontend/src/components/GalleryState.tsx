import type { Person } from "../api/types";

/**
 * Whether a person can actually be matched, and the copy that says so.
 *
 * A person with no template is not a lesser version of an enrolled person: the
 * matcher compares embeddings against templates, so a person with none can
 * never be proposed as a candidate no matter how many times they appear. That
 * is a hole in the gallery rather than a missing field, so it is stated as a
 * state wherever a person is listed, and stated with its remedy on the page
 * the operator would go to in order to fix it.
 */

/** Said on the person page and after an enrolment that produced nothing. */
export const NO_TEMPLATE_EXPLANATION =
  "No template was created, so this person is not in the gallery: matching can never propose them as a candidate.";

/**
 * The way out. Enrolment needs a crop that passed the quality gate, which is a
 * property of the image, so the fix is a better face rather than a retry.
 */
export const NO_TEMPLATE_REMEDY =
  "Tag a sharper or larger face of them — one whose crop passes the quality gate — and confirm it onto this person with “Also add this crop as a template” ticked.";

/**
 * Said when the person once had templates and every one of them is revoked.
 * Distinct from {@link NO_TEMPLATE_EXPLANATION}: nothing failed here, the
 * operator took the face out on purpose, and the rows are still on the page.
 */
export const ALL_TEMPLATES_REVOKED =
  "Every template of this person is revoked, so they are no longer in the gallery: matching can never propose them as a candidate.";

/**
 * Why a duplicate face is worse than untidy, said where the operator can act
 * on it. Two persons holding the same face score the same against it, so the
 * margin rule that guards auto-accept has nothing to separate them and refuses
 * to name either, however high the score is.
 */
export const DUPLICATE_FACE_TIE_EXPLANATION =
  "If another person holds a template of this same face, both score alike on every match and the margin rule refuses to name either, so even a 0.99 match stays ambiguous forever — revoking the duplicate template is what breaks the tie.";

/**
 * Said when a tag was saved but its crop could not be donated. The person may
 * already be in the gallery, so this is about the crop, not about them.
 */
export const NO_CROP_DONATED =
  "The tag was saved, but this crop was not added to the gallery: it did not pass the quality gate.";

/**
 * Enrollment state for a list row or tile.
 *
 * `explain` adds the consequence in words for the layouts that have room; the
 * chip alone would only say "not in gallery", which is a fact without a
 * meaning.
 */
export function GalleryState({ person, explain }: { person: Person; explain: boolean }) {
  if (person.template_count > 0) {
    return <span className={`status-chip status-${person.status}`}>{person.status}</span>;
  }
  return (
    <>
      <span className="status-chip chip-attention">not in gallery</span>
      {explain && <span className="muted">cannot be matched</span>}
    </>
  );
}
