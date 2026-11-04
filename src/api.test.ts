import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchCatalog, fetchProject } from "./api";

afterEach(() => vi.unstubAllGlobals());

describe("project catalog API client", () => {
  it("serializes search, area, and pagination filters", async () => {
    const response = {
      items: [],
      total: 0,
      offset: 12,
      limit: 12,
      areas: [],
    };
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(response), {
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await fetchCatalog({
      q: "text processing",
      area: "Natural Language Processing",
      offset: 12,
      limit: 12,
    });

    const [url] = fetchMock.mock.calls[0] as [string];
    expect(url).toContain("q=text+processing");
    expect(url).toContain("area=Natural+Language+Processing");
    expect(url).toContain("offset=12");
  });

  it("uses the API error detail when a feature is missing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Project not found." }), {
          status: 404,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    await expect(fetchProject("missing")).rejects.toThrow("Project not found.");
  });
});
