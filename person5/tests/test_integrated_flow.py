import os
from pathlib import Path
from uuid import uuid4

os.environ.setdefault(
    "DATABASE_URL", f"sqlite:///{Path(os.getenv('TEMP', '/tmp')) / 'satquery-integration.db'}"
)
os.environ.setdefault("JWT_SECRET_KEY", "integration-test-secret")

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin  # type: ignore[import-untyped]

import person5.backend.analysis as analysis_routes
from person5.backend import database, upload
from person5.backend.main import create_app


def _write_raster(path: Path, bands: int = 3, value: float = 1.0) -> None:
    data = np.full((bands, 8, 8), value, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(10, 10, 1, 1),
    ) as dataset:
        dataset.write(data)


def _client(tmp_path: Path) -> TestClient:
    database.Base.metadata.drop_all(database.engine)
    database.Base.metadata.create_all(database.engine)
    upload.UPLOAD_DIRECTORY = tmp_path / "uploads"
    upload.PROJECT_ROOT = tmp_path
    analysis_routes.PROJECT_ROOT = tmp_path
    return TestClient(create_app())


def _token(client: TestClient, email: str) -> str:
    response = client.post("/auth/register", json={"email": email, "password": "StrongPass123!"})
    assert response.status_code == 201
    return response.json()["access_token"]


def _upload(client: TestClient, token: str, path: Path) -> int:
    response = client.post(
        "/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (path.name, path.read_bytes(), "image/tiff")},
    )
    assert response.status_code == 201
    assert response.json()["file_path"] is None
    return response.json()["id"]


def test_single_image_flow_persists_unavailable_vision_and_trace(tmp_path: Path) -> None:
    client = _client(tmp_path)
    token = _token(client, f"single-{uuid4()}@example.com")
    source = tmp_path / "single.tif"
    _write_raster(source, bands=3)
    image_id = _upload(client, token, source)

    created = client.post(
        "/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"image_id": image_id, "question": "What is visible?"},
    )
    assert created.status_code == 201
    analysis_id = created.json()["analysis_id"]
    assert created.json()["plan"]["task"] == "vqa"

    run = client.post(f"/analyze/{analysis_id}/run", headers={"Authorization": f"Bearer {token}"})
    assert run.status_code == 200
    result = client.get(f"/result/{analysis_id}", headers={"Authorization": f"Bearer {token}"})
    assert result.status_code == 200
    body = result.json()
    assert body["final_result"]["status"] == "partial"
    assert any(
        item["specialist"] == "vision" and item["status"] == "unavailable"
        for item in body["final_result"]["specialist_results"]
    )
    assert body["trace"]["events"]


def test_change_detection_uses_person4_fallback(tmp_path: Path) -> None:
    client = _client(tmp_path)
    token = _token(client, f"change-{uuid4()}@example.com")
    before, after = tmp_path / "before.tif", tmp_path / "after.tif"
    _write_raster(before, value=1.0)
    _write_raster(after, value=2.0)
    ids = [_upload(client, token, before), _upload(client, token, after)]

    created = client.post(
        "/analyze",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "image_ids": ids,
            "question": "What changed?",
            "requested_capability": "change_detection",
        },
    )
    assert created.status_code == 201
    analysis_id = created.json()["analysis_id"]
    assert created.json()["plan"]["task"] == "change_detection"
    assert (
        client.post(
            f"/analyze/{analysis_id}/run", headers={"Authorization": f"Bearer {token}"}
        ).status_code
        == 200
    )
    body = client.get(f"/result/{analysis_id}", headers={"Authorization": f"Bearer {token}"}).json()
    change = next(
        item
        for item in body["final_result"]["specialist_results"]
        if item["specialist"] == "change_detection"
    )
    assert change["status"] == "completed"
    assert "deterministic" in " ".join(change["limitations"])
    
    import json
    answer = json.loads(change["answer"])
    assert "mask_key" in answer
    assert "bounds" in answer

    mask_response = client.get(f"/result/{analysis_id}/mask", headers={"Authorization": f"Bearer {token}"})
    assert mask_response.status_code == 200
    assert mask_response.headers["content-type"] == "image/png"


def test_optical_sar_and_ownership_are_explicit(tmp_path: Path) -> None:
    client = _client(tmp_path)
    token = _token(client, f"fusion-{uuid4()}@example.com")
    optical, sar = tmp_path / "optical.tif", tmp_path / "sar.tif"
    _write_raster(optical, bands=3, value=1.0)
    _write_raster(sar, bands=1, value=2.0)
    ids = [_upload(client, token, optical), _upload(client, token, sar)]
    created = client.post(
        "/analyze",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "image_ids": ids,
            "question": "Compare the optical and SAR observations.",
            "requested_capability": "optical_sar_fusion",
        },
    )
    assert created.status_code == 201
    analysis_id = created.json()["analysis_id"]
    assert (
        client.post(
            f"/analyze/{analysis_id}/run", headers={"Authorization": f"Bearer {token}"}
        ).status_code
        == 200
    )
    body = client.get(f"/result/{analysis_id}", headers={"Authorization": f"Bearer {token}"}).json()
    assert any(
        item["specialist"] == "optical_sar" and item["status"] == "completed"
        for item in body["final_result"]["specialist_results"]
    )
    assert any(
        item["specialist"] == "fusion" for item in body["final_result"]["specialist_results"]
    )

    other = _token(client, f"other-{uuid4()}@example.com")
    forbidden = client.get(f"/result/{analysis_id}", headers={"Authorization": f"Bearer {other}"})
    assert forbidden.status_code == 404
