import type { AnalysisCreateResponse, AnalysisRequest, AnalysisResult, AuthResponse, ExecutionRecord, HealthResponse, UploadedImage } from "./contracts";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const tokenKey = "satquery.access_token";
const requestTimeoutMs = 15_000;

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); this.name = "ApiError"; }
}

export class ApiNetworkError extends Error {
  constructor(public kind: "timeout" | "unreachable") {
    super(kind === "timeout" ? "The backend did not respond before the request timed out." : "The backend could not be reached.");
    this.name = "ApiNetworkError";
  }
}

function authHeaders(): HeadersInit {
  const token = localStorage.getItem(tokenKey);
  return token ? { Authorization: `Bearer ${token}` } : {};
}

const wait = (milliseconds: number) => new Promise<void>((resolve) => globalThis.setTimeout(resolve, milliseconds));

async function fetchWithRetry(path: string, options: RequestInit): Promise<Response> {
  const retryableMethod = (options.method ?? "GET").toUpperCase() === "GET";
  for (let attempt = 0; ; attempt += 1) {
    const controller = new AbortController();
    const timeout = globalThis.setTimeout(() => controller.abort(), requestTimeoutMs);
    try {
      const response = await fetch(`${API_BASE_URL}${path}`, {
        ...options,
        signal: controller.signal,
        headers: { ...authHeaders(), ...options.headers },
      });
      if (!retryableMethod || ![502, 503, 504].includes(response.status) || attempt === 2) return response;
    } catch (error) {
      if (!retryableMethod || attempt === 2) {
        throw new ApiNetworkError(controller.signal.aborted ? "timeout" : "unreachable");
      }
    } finally {
      globalThis.clearTimeout(timeout);
    }
    await wait(250 * (attempt + 1));
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetchWithRetry(path, options);
  if (!response.ok) {
    if (response.status === 401) { clearToken(); if (typeof window !== "undefined") window.dispatchEvent(new Event("satquery:unauthorized")); }
    let message = `Request failed (${response.status})`;
    try { const body = await response.json() as { detail?: string }; if (body.detail) message = body.detail; } catch { /* response may not be JSON */ }
    throw new ApiError(response.status, message);
  }
  return response.json() as Promise<T>;
}

export function getToken(): string | null { return localStorage.getItem(tokenKey); }
export function setToken(token: string): void { localStorage.setItem(tokenKey, token); }
export function clearToken(): void { localStorage.removeItem(tokenKey); }
export function fetchHealth(): Promise<HealthResponse> { return request<HealthResponse>("/health"); }

export const authApi = {
  register: async (email: string, password: string) => request<AuthResponse>("/auth/register", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password }) }),
  login: async (email: string, password: string) => { const body = new URLSearchParams({ username: email, password }); return request<AuthResponse>("/auth/login", { method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" }, body }); },
};

export const analysisApi = {
  upload: (file: File) => { const formData = new FormData(); formData.append("file", file); return request<UploadedImage>("/upload", { method: "POST", body: formData }); },
  createQuery: (payload: AnalysisRequest) => request<AnalysisCreateResponse>("/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }),
  createAnalysis: (payload: AnalysisRequest) => request<AnalysisCreateResponse>("/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }),
  run: (id: number) => request<AnalysisCreateResponse>(`/analyze/${id}/run`, { method: "POST" }),
  result: (id: number) => request<AnalysisResult>(`/result/${id}`),
  execution: (id: number) => request<ExecutionRecord[]>(`/execution/${id}`),
  getMask: async (id: number) => {
    const response = await fetchWithRetry(`/result/${id}/mask`, {});
    if (!response.ok) throw new ApiError(response.status, `Failed to fetch mask (${response.status})`);
    const blob = await response.blob();
    return URL.createObjectURL(blob);
  }
};
