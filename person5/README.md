PERSON 5 — BACKEND + INTEGRATION

WORK:
Connect the entire SatQuery system.

Connect:
- Person 1 Agent
- Person 2 Satellite Data
- Person 3 Vision AI
- Person 4 Change/SAR AI
- Person 6 Frontend

Handle:
- Image uploads
- User questions
- AI analysis requests
- Results
- Database
- Communication between components

TOOLS:
- Python
- FastAPI
- PostgreSQL
- PostGIS
- Redis
- Docker

FOLDERS:

backend/
Main FastAPI backend and API endpoints.

integration/
Connections between the different team components.

tests/
Backend and full-system integration tests.

## Integrated API

Run `person5.backend.main:app` after setting `DATABASE_URL` and `JWT_SECRET_KEY`.
The authenticated API is:

- `POST /auth/register` and `POST /auth/login`
- `POST /upload`
- `POST /query` or `POST /analyze` with `image_id` or ordered `image_ids`, `question`, and optional `requested_capability`
- `POST /analyze/{analysis_id}/run`
- `GET /result/{analysis_id}`
- `GET /execution/{analysis_id}`
- `GET /health`

The execution service converts database images to shared `ImageAsset` contracts, asks Person 1 for a plan, executes providers through `integration/adapters.py`, then persists the exact request, plan, specialist results, final synthesis, and execution trace in the existing SQLAlchemy tables. `SATQUERY_VISION_MODEL` enables the local Person 3 model provider; `SATQUERY_BIT_CHECKPOINT` enables the Person 4 BIT checkpoint. Without those settings, results remain truthful and explicitly unavailable or fallback-labeled.

FINAL OUTPUT:
One backend through which the frontend can communicate with the entire SatQuery AI system.