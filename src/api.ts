export type Project = {
  id: string;
  title: string;
  area: string;
  category: string;
  description: string;
  source: string;
  legacy_module: string;
  legacy_node: string | null;
  legacy_view: string | null;
};

export type CatalogResponse = {
  items: Project[];
  total: number;
  offset: number;
  limit: number;
  areas: string[];
};

export type CatalogQuery = {
  q?: string;
  area?: string;
  sort?: "title" | "area" | "category";
  offset?: number;
  limit?: number;
};

const API_TIMEOUT_MS = 8_000;

export async function fetchCatalog(
  query: CatalogQuery = {},
  signal?: AbortSignal,
): Promise<CatalogResponse> {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== "") params.set(key, String(value));
  }
  return request<CatalogResponse>(
    `/api/catalog${params.size > 0 ? `?${params.toString()}` : ""}`,
    signal,
  );
}

export function fetchProject(id: string, signal?: AbortSignal): Promise<Project> {
  return request<Project>(`/api/catalog/${encodeURIComponent(id)}`, signal);
}

async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  const timeout = AbortSignal.timeout(API_TIMEOUT_MS);
  const combinedSignal = signal ? AbortSignal.any([signal, timeout]) : timeout;
  let response: Response;
  try {
    response = await fetch(path, {
      headers: { Accept: "application/json" },
      signal: combinedSignal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "TimeoutError") {
      throw new Error("The API request timed out. Check the backend and retry.", {
        cause: error,
      });
    }
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error("Could not reach the API. Start the backend and retry.", {
      cause: error,
    });
  }

  if (!response.ok) {
    let message = `The API returned an error (${response.status}).`;
    try {
      const body: unknown = await response.json();
      if (
        typeof body === "object" &&
        body !== null &&
        "detail" in body &&
        typeof body.detail === "string"
      ) {
        message = body.detail;
      }
    } catch {
      // Retain the HTTP status message if the server did not return JSON.
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}
