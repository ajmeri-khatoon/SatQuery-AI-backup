export type SpecialistStatus = "completed" | "failed" | "unavailable";
export type AnalysisStatus = "pending" | "running" | "completed" | "partial" | "failed" | "unavailable" | "rejected";
export type AnalysisMode = "single" | "change" | "optical-sar";
export type AssetSlot = "single" | "before" | "after" | "optical" | "sar";
export type RequestedCapability = "auto" | "vqa" | "caption" | "grounding" | "change_detection" | "optical_sar_fusion";

export interface HealthResponse { status: "ok"; stage: string; providers: string; }
export interface AuthResponse { access_token: string; token_type: string; }
export interface UploadedImage { id: number; filename: string; file_path: null; created_at: string; }
export interface UploadedAsset extends UploadedImage { slot: AssetSlot; status: "uploading" | "ready" | "failed"; error?: string; size: number; format: string; previewUrl?: string; }

export interface AnalysisRequest { image_id?: number; image_ids?: number[]; question: string; requested_capability: RequestedCapability; }
export interface PlanStep { id: string; specialist: string; operation: string; input_asset_ids: string[]; depends_on: string[]; status: string; }
export interface TaskPlan { analysis_id: string; contract_version: string; task: string; specialists: string[]; steps: PlanStep[]; validation: string; limitations: string[]; }
export interface SpecialistResult { analysis_id: string; step_id: string; specialist: string; status: SpecialistStatus; answer: string | null; confidence: number | null; confidence_method?: string | null; evidence_ids: string[]; limitations: string[]; provenance: Record<string, string>; error_code?: string | null; }
export interface ResultItem { id: number; result_type: string | null; data: SpecialistResult | Record<string, unknown> | null; created_at: string; }
export interface TraceEvent { event_id: string; timestamp: string; step_id: string; event_type: string; component: string; message: string; details: Record<string, string | number | boolean | null>; }
export interface ExecutionTrace { analysis_id: string; trace_id: string; started_at: string; finished_at: string | null; events: TraceEvent[]; outcome: "completed" | "failed" | "partial" | "rejected"; }
export interface AnalysisCreateResponse { analysis_id: number; status: string; plan: TaskPlan | null; }
export interface AnalysisResult { analysis_id: number; image_id: number | null; question: string | null; status: AnalysisStatus; results: ResultItem[]; plan: TaskPlan | null; final_result: { status: string; answer: string | null; confidence: number | null; evidence_ids: string[]; limitations: string[]; provenance: Record<string, string>; specialist_results: SpecialistResult[]; trace: ExecutionTrace } | null; trace: ExecutionTrace | null; }
export interface ExecutionRecord { id: number; step: string; specialist: string | null; status: string; started_at: string | null; completed_at: string | null; error_message: string | null; created_at: string; result_data: SpecialistResult | null; }
