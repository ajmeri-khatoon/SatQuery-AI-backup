import { describe, expect, it } from "vitest";
import { resultMessage } from "../src/resultPresentation";

describe("resultMessage", () => {
  it("does not present unavailable output as a real answer", () => {
    expect(resultMessage({ analysis_id: "analysis", step_id: "vision", specialist: "vision", status: "unavailable", answer: null, confidence: null, evidence_ids: [], provenance: {}, limitations: [] })).toContain("not performed");
  });
});
