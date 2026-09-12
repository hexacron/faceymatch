import { useState } from "react";

import type { Appearance, PersonDetail } from "../api/types";
import { BandPill, SourceBadge } from "../components/BandPill";
import {
  DoNotEnrollToggle,
  RevokeTemplateControl,
  type GalleryNotice,
} from "../components/GalleryControls";
import {
  ALL_TEMPLATES_REVOKED,
  DUPLICATE_FACE_TIE_EXPLANATION,
  GalleryState,
  NO_TEMPLATE_EXPLANATION,
  NO_TEMPLATE_REMEDY,
} from "../components/GalleryState";
import { Loaded } from "../components/Loading";
import { formatTs, truncateHash } from "../lib/display";
import { hrefFor } from "../lib/router";
import { useResource } from "../lib/useResource";

function AppearanceCard({ appearance }: { appearance: Appearance }) {
  return (
    <article className="appearance-card">
      {appearance.crop_sha256 === null ? (
        <div className="crop-missing">No crop</div>
      ) : (
        <img
          src={`/api/crops/${encodeURIComponent(appearance.crop_sha256)}`}
          alt={`Appearance crop ${truncateHash(appearance.crop_sha256)}`}
        />
      )}
      <div>
        <a href={hrefFor({ view: "viewer", mediaId: appearance.media_id })}>Open media</a>
        <p className="mono compact">{formatTs(appearance.ingested_at)}</p>
        <BandPill band={appearance.band} score={appearance.score} />{" "}
        <SourceBadge source={appearance.source} />
        <p className="muted compact">at {String(appearance.t_ms)} ms · case <span className="mono">{truncateHash(appearance.case_id)}</span></p>
      </div>
    </article>
  );
}

export default function PersonDetailView({ personId }: { personId: string }) {
  const detail = useResource<PersonDetail>(`/api/persons/${encodeURIComponent(personId)}`);
  /**
   * One place for the outcome of a revoke or a park. It lives above the cards
   * because the control that produced it disappears the moment the change
   * lands — the template stops being active, so there is nothing left to
   * revoke and nowhere for the message to sit.
   */
  const [notice, setNotice] = useState<GalleryNotice | null>(null);

  return (
    <>
      <div className="view-heading">
        <div>
          <a className="back-link" href={hrefFor({ view: "persons" })}>← Persons</a>
          <h1>Person detail</h1>
          <p className="tagline mono">{personId}</p>
        </div>
        <button type="button" onClick={detail.reload}>Refresh</button>
      </div>
      <Loaded state={detail.state} label="person details">
        {(data) => {
          const appearances = [...data.appearances].sort((left, right) =>
            left.ingested_at.localeCompare(right.ingested_at),
          );
          // The matcher only ever sees active templates, so this is what
          // "in the gallery" means on this page.
          const activeCount = data.templates.filter((item) => item.status === "active").length;
          return (
            <>
              <section className="panel person-summary" aria-labelledby="person-name">
                <div>
                  <h2 id="person-name" className="title-case">{data.person.display_name}</h2>
                  <p>{data.person.notes ?? "No notes."}</p>
                  <DoNotEnrollToggle
                    person={data.person}
                    onOutcome={setNotice}
                    onRefresh={detail.reload}
                  />
                </div>
                <dl className="facts">
                  <dt>Status</dt><dd className="state-cell"><GalleryState person={data.person} explain={true} /></dd>
                  <dt>Templates</dt><dd>{data.person.template_count}</dd>
                  <dt>Created</dt><dd className="mono">{formatTs(data.person.created_at)}</dd>
                  <dt>Created by</dt><dd>{data.person.created_by}</dd>
                  <dt>Enrollment</dt><dd>{data.person.do_not_enroll ? "Do not enroll" : "Allowed"}</dd>
                </dl>
              </section>

              {notice !== null && (
                <p
                  className={`notice ${notice.tone}`}
                  role={notice.tone === "error" ? "alert" : "status"}
                >
                  {notice.text}
                </p>
              )}

              <section aria-labelledby="templates-heading">
                <h2 id="templates-heading">Templates</h2>
                <p className="notice">{DUPLICATE_FACE_TIE_EXPLANATION}</p>
                {activeCount === 0 && (
                  <p className="notice attention">
                    {data.templates.length === 0 ? NO_TEMPLATE_EXPLANATION : ALL_TEMPLATES_REVOKED}{" "}
                    {NO_TEMPLATE_REMEDY}
                  </p>
                )}
                {data.templates.length > 0 && (
                  <div className="template-grid">
                    {data.templates.map((template) => (
                      <article
                        className={`panel template-card${template.status === "revoked" ? " template-revoked" : ""}`}
                        key={template.id}
                      >
                        {template.crop_sha256 === null ? (
                          <div className="crop-missing">No crop</div>
                        ) : (
                          <img src={`/api/crops/${encodeURIComponent(template.crop_sha256)}`} alt={`Template crop ${truncateHash(template.crop_sha256)}`} />
                        )}
                        <div>
                          <span className={`status-chip status-${template.status}`}>{template.status}</span>
                          <p className="mono compact" title={template.id}>{truncateHash(template.id)}</p>
                          <p className="compact">Quality {template.quality?.toFixed(3) ?? "—"}</p>
                          <p className="muted compact">{template.embedder_model_id}</p>
                        </div>
                        {template.status === "active" ? (
                          <RevokeTemplateControl
                            personId={data.person.id}
                            template={template}
                            onOutcome={setNotice}
                            onRefresh={detail.reload}
                          />
                        ) : (
                          <p className="template-actions muted compact">
                            Out of the matching gallery. The row, the crop and the audit trail are
                            kept; revoking cannot be undone, so re-enrol this face to use it again.
                          </p>
                        )}
                      </article>
                    ))}
                  </div>
                )}
              </section>

              <section aria-labelledby="appearances-heading">
                <h2 id="appearances-heading">Appearances</h2>
                {appearances.length === 0 ? (
                  <p className="notice">No accepted appearances.</p>
                ) : (
                  <div className="appearance-grid">
                    {appearances.map((appearance) => <AppearanceCard key={appearance.track_id} appearance={appearance} />)}
                  </div>
                )}
              </section>

              <section aria-labelledby="timeline-heading">
                <h2 id="timeline-heading">Timeline</h2>
                {appearances.length === 0 ? (
                  <p className="notice">The timeline is empty.</p>
                ) : (
                  <ol className="timeline">
                    {appearances.map((appearance) => (
                      <li key={appearance.track_id}>
                        <time dateTime={appearance.ingested_at}>{formatTs(appearance.ingested_at)}</time>
                        <a href={hrefFor({ view: "viewer", mediaId: appearance.media_id })}>Media {truncateHash(appearance.media_id)}</a>
                        <SourceBadge source={appearance.source} />
                      </li>
                    ))}
                  </ol>
                )}
              </section>
            </>
          );
        }}
      </Loaded>
    </>
  );
}
