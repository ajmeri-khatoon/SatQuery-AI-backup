# SATQUERY Project Constitution

## Mission

SATQUERY AI is the Smart India Hackathon 2026 prototype for **SIH26167 — SatQuery AI: An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**. It lets users ask grounded natural-language questions about satellite imagery. It must be scientifically honest: it is a remote-sensing system assisted by AI, not a generic chatbot that claims to analyze imagery.

## Product and architecture

```text
React / MapLibre GIS frontend
  -> Person 5 FastAPI integration backend
  -> Person 1 query planner and orchestrator
  -> Person 2 ingestion | Person 3 vision | Person 4 change / optical-SAR specialists
  -> fused result, evidence, confidence, provenance, execution trace, persistence
```

Required capabilities are: optical and multispectral single-image analysis; SAR analysis; co-registered optical/SAR analysis; before/after change analysis; satellite/GeoTIFF ingestion; metadata, geospatial alignment, and tiling; remote-sensing VQA, captioning, and grounding; and GIS-visible evidence where available. Specialist processing and models must be appropriate to the requested capability; a generic vision-language model alone is not sufficient.

## Six-person ownership boundaries

| Owner | Scope |
| --- | --- |
| Person 1 (`person1/`) | Query planning, specialist orchestration, result fusion, confidence, execution trace. |
| Person 2 (`person2/`) | Sentinel and GeoTIFF ingestion, metadata, alignment, tiling, preprocessing. |
| Person 3 (`person3/`) | Remote-sensing VQA, captioning, grounding, model/dataset adaptation. |
| Person 4 (`person4/`) | Change detection; optical/SAR analysis and fusion. |
| Person 5 (`person5/`) | FastAPI, uploads, integration, persistence, API and execution tracking. |
| Person 6 (`person6/`) | React UI and MapLibre/GIS evidence visualisation. |

Preserve human and team implementations. Inspect the existing implementation, contracts, tests, and owner documentation before proposing or making a change. Do not replace a teammate's module to make integration easier: extend it through an adapter or a versioned shared contract. In particular, reuse the existing Person 5 backend as the integration backend rather than rebuilding it, and reuse an existing team frontend when supplied rather than replacing it.

## Shared contracts and truthful results

Shared, typed, versioned contracts are the integration boundary. Validate at every boundary; make new fields backward-compatible where possible; and keep provider-specific details behind adapters. Coordinate contract changes with all affected owners.

Every result must clearly carry, as applicable:

- the answer and its supported/calibrated confidence;
- visual or geospatial evidence and the source imagery/region/time inputs;
- model, processing, dataset, and configuration provenance;
- limitations, uncertainty, assumptions, and unavailable components; and
- an auditable execution trace showing planner decisions, specialist calls, inputs, outputs, failures, and fusion.

Never fabricate analysis, confidence, evidence, provenance, execution records, or a successful specialist call. If a required model, dataset, credential, image, provider, or processing capability is unavailable, return an explicit unavailable/unsupported result with limitations and a trace entry. Label prototype, heuristic, stub, mock, and demo functionality prominently; do not represent it as genuine remote-sensing inference. Genuine claims require real applicable processing/models and source-backed evidence.

## Agentic orchestration

The orchestrator must select only suitable specialists, record its routing and fusion decisions, propagate uncertainty and failures, and avoid unsupported conclusions. Use the smallest coordination approach that fits the task; coordination tools record work but never replace implementation, validation, or ownership review. No two writers may edit the same owned scope concurrently. The integration owner alone changes shared manifests or lockfiles.

## Security, validation, and operational limits

- Never commit or hardcode credentials, tokens, `.env` files, model weights, or large datasets. Use local configuration and validate all external input, upload, and file paths.
- Do not add infrastructure, services, APIs, dependencies, models, or datasets unless they are necessary for an approved capability. Prefer the existing application and provider boundaries.
- For changes, run focused validation proportionate to risk: contract/type checks and targeted unit or integration tests, including unavailable-provider and failure-path behaviour. Do not claim validation that was not run.
- Do not run destructive Git or remote operations — including reset, clean, checkout/revert, force-push, merge, release, or deployment — unless explicitly authorized. Do not auto-commit or push.

## Working rule

Before changing code, inspect what exists, identify the owning boundary, preserve compatible behaviour, and make the smallest change that advances the real remote-sensing product. Do not modify application or frontend code unless the requested work requires it and the relevant ownership is respected.
