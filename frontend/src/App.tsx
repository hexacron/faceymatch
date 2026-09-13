import { IngestNotices } from "./components/IngestNotices";
import ConfigView from "./views/ConfigView";
import LiveView from "./views/LiveView";
import MediaDetailView from "./views/MediaDetailView";
import MediaLibraryView from "./views/MediaLibraryView";
import PersonDetailView from "./views/PersonDetailView";
import PersonsView from "./views/PersonsView";
import ReviewView from "./views/ReviewView";
import StatusView from "./views/StatusView";
import { useGlobalIngest } from "./lib/ingest";
import { hrefFor, useRoute, type Route } from "./lib/router";

type NavItem = { label: string; route: Route; activeViews: readonly Route["view"][] };

const NAV_ITEMS: readonly NavItem[] = [
  { label: "Media", route: { view: "media" }, activeViews: ["media", "viewer"] },
  { label: "Live", route: { view: "live" }, activeViews: ["live"] },
  { label: "Persons", route: { view: "persons" }, activeViews: ["persons", "person"] },
  { label: "Review", route: { view: "review" }, activeViews: ["review"] },
  { label: "Status", route: { view: "status" }, activeViews: ["status"] },
  { label: "Config", route: { view: "config" }, activeViews: ["config"] },
];

function CurrentView({ route }: { route: Route }) {
  switch (route.view) {
    case "media":
      return <MediaLibraryView />;
    case "viewer":
      return <MediaDetailView mediaId={route.mediaId} focus={route.focus ?? null} />;
    case "live":
      return <LiveView />;
    case "persons":
      return <PersonsView />;
    case "person":
      // Keyed on the id: the page holds state that is only true of one person —
      // notably the terminal "deleted" panel — and reusing the instance for the
      // next person would show them somebody else's outcome.
      return <PersonDetailView key={route.personId} personId={route.personId} />;
    case "review":
      return <ReviewView />;
    case "status":
      return <StatusView />;
    case "config":
      return <ConfigView />;
  }
}

export default function App() {
  const route = useRoute();
  // Cmd+V anywhere ingests a clipboard image into the selected case.
  useGlobalIngest();
  return (
    <>
      <a className="skip-link" href="#main-content">Skip to content</a>
      <header className="app-header">
        <a className="brand" href={hrefFor({ view: "media" })}>faceymatch</a>
        <span className="app-subtitle">Local operator console</span>
        <nav aria-label="Primary navigation">
          {NAV_ITEMS.map((item) => (
            <a
              key={item.label}
              href={hrefFor(item.route)}
              aria-current={item.activeViews.includes(route.view) ? "page" : undefined}
            >
              {item.label}
            </a>
          ))}
        </nav>
      </header>
      <main className="shell" id="main-content">
        <CurrentView route={route} />
      </main>
      <IngestNotices />
    </>
  );
}
