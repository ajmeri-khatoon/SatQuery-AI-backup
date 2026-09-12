# SATQUERY AI

SATQUERY AI is a Smart India Hackathon 2026 prototype for SIH26167: **SatQuery AI — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**.

Users will be able to provide satellite imagery and ask natural-language questions about it. The system is being built to handle single optical or multispectral images, SAR imagery, co-registered optical/SAR pairs, and before/after change analysis.

## Product principles

- Remote-sensing outputs must be produced by real, appropriate specialist processing or models; a generic vision-language model alone is not sufficient.
- Dataset and model adaptation will be grounded in suitable open remote-sensing resources, such as BigEarthNet, VRSBench, RSVQA, CDVQA, or Open-CD, as applicable to each capability.
- Results must include calibrated confidence where supported, visual or geospatial evidence where available, provenance, limitations, and an auditable execution trace.
- An unavailable specialist must report that it is unavailable. SATQUERY never presents fabricated AI output as analysis.

## Architecture

```text
Web GIS frontend
  -> FastAPI backend and analysis manager
  -> agent orchestrator
  -> preprocessing / vision / change / optical-SAR specialists
  -> result fusion, evidence, confidence, and execution trace
  -> persistent storage
```

The frontend is TypeScript/React with GIS visualization. The backend and specialist modules are Python. Shared, versioned contracts define the boundary between them.

## Team ownership

| Area | Responsibility |
| --- | --- |
| `person1/` | Query planning, specialist orchestration, fusion, confidence, trace |
| `person2/` | Sentinel and GeoTIFF ingestion, metadata, alignment, tiling |
| `person3/` | Remote-sensing VQA, captioning, grounding, dataset/model adaptation |
| `person4/` | Change detection and optical/SAR analysis and fusion |
| `person5/` | FastAPI, uploads, integration, persistence, execution tracking |
| `person6/` | React UI, MapLibre/GIS evidence visualization |

## Current stage

This backup repository is establishing a reproducible application foundation. It does not yet include downloaded models, datasets, or completed remote-sensing inference. The initial foundation provides validated contracts, safe unavailable-provider behavior, module boundaries, and tests before those capabilities are added.

## Repository safety

This is a personal backup and AI-development repository. It is intentionally separate from the team repository. No credentials, model weights, or large datasets belong in Git.

## Development

Setup commands will be documented once the Python and frontend manifests are in place. Copy `.env.example` to a local `.env` only when a future phase needs runtime configuration; never commit it.
