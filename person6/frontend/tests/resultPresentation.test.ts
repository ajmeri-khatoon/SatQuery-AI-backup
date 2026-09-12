import { describe, expect, it } from "vitest";
import { resultMessage } from "../src/resultPresentation";

describe("resultMessage", () => {
  it("does not present unavailable output as a real answer", () => {
    expect(resultMessage({ specialist: "vision", status: "unavailable", answer: null, confidence: null, limitations: [] })).toContain("not performed");
  });
});
