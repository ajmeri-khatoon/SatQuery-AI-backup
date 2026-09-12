export type SpecialistStatus = "completed" | "failed" | "unavailable";

export interface SpecialistResult {
  specialist: string;
  status: SpecialistStatus;
  answer: string | null;
  confidence: number | null;
  limitations: string[];
}

export interface HealthResponse {
  status: "ok";
  stage: "foundation";
  providers: "unconfigured";
}
