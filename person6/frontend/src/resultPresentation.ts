import type { SpecialistResult } from "./contracts";

export function resultMessage(result: SpecialistResult): string {
  if (result.status !== "completed") return "Analysis was not performed by this specialist.";
  return result.answer ?? "The specialist completed without a textual answer.";
}
