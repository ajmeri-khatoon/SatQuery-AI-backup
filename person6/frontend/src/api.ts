import type { AnalysisCreateResponse, AnalysisRequest, AnalysisResult, AuthResponse, ExecutionRecord, HealthResponse, UploadedImage } from "./contracts";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";
const tokenKey = "satquery.access_token";

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); this.name = "ApiError"; }
}

function authHeaders(): HeadersInit {
  const token = localStorage.getItem(tokenKey);
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers: { ...authHeaders(), ...options.headers } });
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
    const response = await fetch(`${API_BASE_URL}/result/${id}/mask`, { headers: authHeaders() });
    if (!response.ok) throw new ApiError(response.status, "Failed to fetch mask");
    const blob = await response.blob();
    return URL.createObjectURL(blob);
  }
};
