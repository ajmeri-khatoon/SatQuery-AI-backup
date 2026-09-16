import { beforeEach, describe, expect, it, vi } from "vitest";
import { analysisApi, ApiNetworkError, authApi, fetchHealth, setToken } from "../src/api";

const storage = new Map<string, string>();
const response = (body: unknown, status = 200) => ({ ok: status >= 200 && status < 300, status, json: async () => body });

beforeEach(() => {
  storage.clear();
  vi.stubGlobal("localStorage", { getItem: (key: string) => storage.get(key) ?? null, setItem: (key: string, value: string) => storage.set(key, value), removeItem: (key: string) => storage.delete(key) });
  vi.stubGlobal("fetch", vi.fn());
});

describe("P5 API boundary", () => {
  it("registers and logs in against the real auth routes", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(response({ access_token: "register-token", token_type: "bearer" }) as Response).mockResolvedValueOnce(response({ access_token: "login-token", token_type: "bearer" }) as Response);
    await expect(authApi.register("analyst@example.com", "password123")).resolves.toMatchObject({ access_token: "register-token" });
    await expect(authApi.login("analyst@example.com", "password123")).resolves.toMatchObject({ access_token: "login-token" });
    expect(fetchMock.mock.calls[0][0]).toContain("/auth/register");
    expect(fetchMock.mock.calls[1][0]).toContain("/auth/login");
    expect((fetchMock.mock.calls[1][1]?.body as URLSearchParams).get("username")).toBe("analyst@example.com");
  });

  it("sends authenticated upload, analysis, execution, result, and trace requests", async () => {
    setToken("session-token");
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValue(response({ analysis_id: 42, status: "completed", plan: null }) as Response);
    const file = new File(["raster"], "scene.tif", { type: "image/tiff" });
    await analysisApi.upload(file);
    await analysisApi.createQuery({ image_id: 7, question: "What is visible?", requested_capability: "auto" });
    await analysisApi.run(42);
    await analysisApi.result(42);
    await analysisApi.execution(42);
    const paths = fetchMock.mock.calls.map(([path]) => {
      try { return new URL(String(path)).pathname; }
      catch { return String(path); }
    });
    expect(paths).toEqual(["/upload", "/query", "/analyze/42/run", "/result/42", "/execution/42"]);
    expect((fetchMock.mock.calls[1][1]?.headers as Record<string, string>).Authorization).toBe("Bearer session-token");
  });

  it("surfaces backend error status and detail", async () => {
    vi.mocked(fetch).mockResolvedValue(response({ detail: "Analysis is already running" }, 409) as Response);
    await expect(analysisApi.run(42)).rejects.toMatchObject({ status: 409, message: "Analysis is already running" });
  });

  it("retries a transient backend wake-up response for safe GET requests", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce(response({}, 503) as Response).mockResolvedValueOnce(response({ status: "ok", stage: "integrated", providers: "configured" }) as Response);
    await expect(fetchHealth()).resolves.toMatchObject({ status: "ok" });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("distinguishes an unreachable backend from an HTTP response", async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new TypeError("network failed"));
    await expect(authApi.login("analyst@example.com", "password123")).rejects.toBeInstanceOf(ApiNetworkError);
  });
});
