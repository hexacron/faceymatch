import { useMemo, useState, type FormEvent } from "react";

import type { Person, PersonList, PersonStatus } from "../api/types";
import { GalleryState } from "../components/GalleryState";
import { Loaded } from "../components/Loading";
import { formatTs } from "../lib/display";
import { setPersonsLayout, usePersonsLayout } from "../lib/personsLayout";
import { hrefFor } from "../lib/router";
import { useResource } from "../lib/useResource";

/** Shared by both layouts so a single match does not read "1 persons". */
function countLabel(count: number): string {
  return `${String(count)} person${count === 1 ? "" : "s"} in the global gallery`;
}

/**
 * One face in the gallery.
 *
 * The whole tile is the link, like the name cell in the table, so the target
 * is the face rather than a few characters of text. `crop_sha256` is null for
 * anyone with no active template, and a bundle older than that field sees
 * nothing there at all, so the placeholder is chosen on "is this a string"
 * rather than on a null check.
 */
function PersonCard({ person }: { person: Person }) {
  const crop = person.crop_sha256;
  const label =
    `${person.display_name}: ` +
    (person.template_count === 0
      ? "not in gallery, cannot be matched"
      : `${person.status}, ${String(person.template_count)} template${person.template_count === 1 ? "" : "s"}`) +
    (person.do_not_enroll ? ", do not enroll" : "");
  return (
    <a
      className="person-card"
      href={hrefFor({ view: "person", personId: person.id })}
      aria-label={label}
    >
      {typeof crop === "string" ? (
        <img src={`/api/crops/${encodeURIComponent(crop)}`} alt={`Face crop for ${person.display_name}`} />
      ) : (
        <div className="crop-missing">No crop</div>
      )}
      <div className="person-card-body">
        <strong>{person.display_name}</strong>
        <div className="person-card-facts">
          <GalleryState person={person} explain={false} />
          <span className="muted">
            {person.template_count === 0
              ? "cannot be matched"
              : `${String(person.template_count)} template${person.template_count === 1 ? "" : "s"}`}
          </span>
          {person.do_not_enroll && <span className="pill">do not enroll</span>}
        </div>
      </div>
    </a>
  );
}

export default function PersonsView() {
  const [queryInput, setQueryInput] = useState("");
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<PersonStatus | "">("");
  const layout = usePersonsLayout();
  const path = useMemo(() => {
    const params = new URLSearchParams();
    if (query !== "") {
      params.set("q", query);
    }
    if (status !== "") {
      params.set("status", status);
    }
    const suffix = params.toString();
    return suffix === "" ? "/api/persons" : `/api/persons?${suffix}`;
  }, [query, status]);
  const persons = useResource<PersonList>(path);

  function search(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    setQuery(queryInput.trim());
  }

  return (
    <>
      <div className="view-heading">
        <div>
          <h1>Persons</h1>
          <p className="tagline">Global gallery identities, templates, and appearances.</p>
        </div>
        <button type="button" onClick={persons.reload}>Refresh</button>
      </div>
      <form className="toolbar panel" onSubmit={search}>
        <label>
          Search
          <input value={queryInput} onChange={(event) => setQueryInput(event.currentTarget.value)} placeholder="Display name" />
        </label>
        <label>
          Enrollment status
          <select value={status} onChange={(event) => setStatus(event.currentTarget.value as PersonStatus | "")}>
            <option value="">All</option>
            <option value="enrolled">Enrolled</option>
            <option value="unenrolled">Unenrolled</option>
          </select>
        </label>
        <label>
          Layout
          <select
            value={layout}
            onChange={(event) => setPersonsLayout(event.currentTarget.value === "list" ? "list" : "tiles")}
          >
            <option value="tiles">Tiles</option>
            <option value="list">List</option>
          </select>
        </label>
        <button type="submit">Search</button>
      </form>
      <Loaded state={persons.state} label="persons">
        {(list) =>
          list.items.length === 0 ? (
            <p className="notice">No persons match these filters.</p>
          ) : layout === "tiles" ? (
            <>
              <p className="muted compact">{countLabel(list.items.length)}</p>
              <div className="person-grid">
                {list.items.map((person) => (
                  <PersonCard key={person.id} person={person} />
                ))}
              </div>
            </>
          ) : (
            <div className="table-scroll panel">
              <table>
                <caption>{countLabel(list.items.length)}</caption>
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Status</th>
                    <th className="num">Templates</th>
                    <th>Created</th>
                    <th>Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {list.items.map((person) => (
                    <tr key={person.id}>
                      <td><a href={hrefFor({ view: "person", personId: person.id })}>{person.display_name}</a></td>
                      <td className="state-cell"><GalleryState person={person} explain /></td>
                      <td className="num">{person.template_count}</td>
                      <td className="mono">{formatTs(person.created_at)}</td>
                      <td>{person.do_not_enroll ? <span className="pill">do not enroll</span> : (person.notes ?? "—")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        }
      </Loaded>
    </>
  );
}
