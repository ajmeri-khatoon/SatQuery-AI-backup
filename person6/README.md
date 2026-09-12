# Person 6: SatQuery frontend

The production-oriented React/TypeScript workstation lives in `frontend/`. It is a desktop-first, responsive analysis shell for SIH26167 connected to Person 5's authenticated API. It deliberately does not fabricate provider output, imagery URLs, coordinates, or confidence.

## Architecture

- `src/components/`: persistent shell, panels, controls, file dropzone, and MapLibre-ready viewer.
- `src/pages.tsx`: route-level views for overview, workspace, data, history, comparison, optical/SAR, results, settings, and auth.
- `src/contracts.ts`: typed frontend boundary for assets, analysis requests, records, results, and existing health contracts.
- `src/api.ts`: isolated auth, upload, query/analyze, execution, result, and trace service functions. Set `VITE_API_BASE_URL` at build time for the Person 5 API.
- `src/styles.css`: restrained scientific workstation visual system with responsive and reduced-motion states.

## Routes

`/`, `/workspace`, `/data`, `/analysis`, `/analysis/:id`, `/compare`, `/optical-sar`, `/settings`, `/login`, and `/register`.

## Development

```bash
npm install
npm run dev
npm run typecheck
npm test
npm run build
```

## Integration status

The frontend connects `POST /auth/register`, `POST /auth/login`, `POST /upload`, `POST /query`, `POST /analyze`, `POST /analyze/{analysis_id}/run`, `GET /result/{analysis_id}`, `GET /execution/{analysis_id}`, and `GET /health`. Tokens are stored for the authenticated session and cleared on 401 or logout. Uploads retain the numeric P5 image ID and analysis creation uses those IDs.

P5 execution is synchronous, so the workspace displays the real request/run states and the result page loads both the final result and execution records. P5 has no history/list endpoint, so history remains an explicit empty state. P5 currently returns no raster URLs or geographic evidence; change masks, optical/SAR layers, and MapLibre remain unavailable unless those fields are added to a future backend contract. Returned specialist JSON, confidence, limitations, provenance, regions, and trace events are rendered as-is.