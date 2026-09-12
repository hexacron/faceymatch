import type { ReactNode } from "react";

import type { Loadable } from "../lib/useResource";

/**
 * Render a loadable resource. Keeps the three phases in one place so every
 * view reports a backend failure the same way instead of silently rendering
 * an empty table.
 */
export function Loaded<T>({
  state,
  children,
  label,
}: {
  state: Loadable<T>;
  label: string;
  children: (data: T) => ReactNode;
}) {
  if (state.phase === "loading") {
    return <p className="notice">Loading {label}&hellip;</p>;
  }
  if (state.phase === "error") {
    return (
      <div className="notice error">
        Could not load {label}: <span className="mono">{state.message}</span>
      </div>
    );
  }
  return <>{children(state.data)}</>;
}
