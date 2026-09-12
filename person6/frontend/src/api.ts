import type { HealthResponse } from "./contracts";

export async function fetchHealth(baseUrl = "http://localhost:8000"): Promise<HealthResponse> {
  const response = await fetch(`${baseUrl}/health`);
  if (!response.ok) throw new Error(`Health request failed: ${response.status}`);
  return response.json() as Promise<HealthResponse>;
}
