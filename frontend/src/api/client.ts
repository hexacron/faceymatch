/**
 * Thin fetch wrapper for the local backend.
 *
 * The backend serves this bundle at `/` and the API under `/api`, so every
 * request stays relative. Never hardcode an origin: in dev Vite proxies `/api`
 * to 127.0.0.1:8000, and in production there is no second origin to reach
 * (invariant 1: no outbound network calls).
 */

export class ApiError extends Error {
  readonly status: number;
  /** The `detail` text alone, for callers that branch on a documented reason. */
  readonly detail: string;

  constructor(status: number, path: string, detail: string) {
    super(`${path}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Pydantic validation failures (422) arrive as a list of
 * `{loc, msg, input, ...}`, and `input` is a copy of what was rejected — a
 * 2000-character basis dumped back into a notice tells the operator nothing.
 * Reduce each entry to "field: message"; return null when the shape is not the
 * documented one, so the caller falls back to the raw JSON rather than lying.
 */
function validationDetail(items: readonly unknown[]): string | null {
  const parts: string[] = [];
  for (const item of items) {
    if (typeof item !== "object" || item === null || !("msg" in item)) {
      return null;
    }
    const message: unknown = item.msg;
    if (typeof message !== "string") {
      return null;
    }
    const location: unknown = "loc" in item ? item.loc : undefined;
    // `loc` starts with the request part ("body", "query"), which names nothing
    // the operator can see.
    const field = Array.isArray(location)
      ? location
          .filter((piece): piece is string => typeof piece === "string" && piece !== "body")
          .join(".")
      : "";
    parts.push(field === "" ? message : `${field}: ${message}`);
  }
  return parts.length === 0 ? null : parts.join("; ");
}

/**
 * FastAPI reports failures as `{"detail": ...}`; surface that text so the
 * operator sees the real reason (a trigger rejection, a license gate) instead
 * of a bare status code.
 */
async function failureFor(path: string, response: Response): Promise<ApiError> {
  let detail = `${response.status} ${response.statusText}`;
  try {
    const body: unknown = await response.json();
    if (typeof body === "object" && body !== null && "detail" in body) {
      const raw: unknown = body.detail;
      if (typeof raw === "string") {
        detail = raw;
      } else {
        detail = (Array.isArray(raw) ? validationDetail(raw) : null) ?? JSON.stringify(raw);
      }
    }
  } catch {
    // Non-JSON error body (a proxy page, an empty 502): keep the status line.
  }
  return new ApiError(response.status, path, detail);
}

export async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, {
    ...(signal === undefined ? {} : { signal }),
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    throw await failureFor(path, response);
  }
  // `Response.json()` is typed `Promise<any>` by the DOM lib; we assert the
  // documented wire type here so the rest of the app stays fully typed.
  return (await response.json()) as T;
}

export async function postJson<T>(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    ...(signal === undefined ? {} : { signal }),
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw await failureFor(path, response);
  }
  return (await response.json()) as T;
}

export async function patchJson<T>(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(path, {
    method: "PATCH",
    ...(signal === undefined ? {} : { signal }),
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw await failureFor(path, response);
  }
  return (await response.json()) as T;
}

/** DELETE, answering with the summary of what went. Used by the section 12 purges. */
export async function deleteJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, {
    method: "DELETE",
    ...(signal === undefined ? {} : { signal }),
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    throw await failureFor(path, response);
  }
  return (await response.json()) as T;
}

export async function postForm<T>(
  path: string,
  form: FormData,
  signal?: AbortSignal,
): Promise<T> {
  // No Content-Type header: the browser has to add the multipart boundary.
  const response = await fetch(path, {
    method: "POST",
    ...(signal === undefined ? {} : { signal }),
    headers: { Accept: "application/json" },
    body: form,
  });
  if (!response.ok) {
    throw await failureFor(path, response);
  }
  return (await response.json()) as T;
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
