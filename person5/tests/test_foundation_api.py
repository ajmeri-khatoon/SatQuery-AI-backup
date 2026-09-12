from fastapi.testclient import TestClient

from person5.backend import create_app
from shared.contracts import AssetFormat, ImageAsset, ImageRole


def payload(role: ImageRole = ImageRole.SINGLE) -> dict:
    asset = ImageAsset(
        original_filename="scene.tif",
        storage_key="assets/scene.tif",
        content_type="image/tiff",
        format=AssetFormat.TIFF,
        role=role,
    )
    return {
        "request": {"question": "What is visible?", "asset_ids": [str(asset.id)]},
        "assets": [asset.model_dump(mode="json")],
    }


def test_health_does_not_require_configuration() -> None:
    response = TestClient(create_app()).get("/health")
    assert response.status_code == 200
    assert response.json()["providers"] == "unconfigured"


def test_execute_returns_explicitly_unavailable_results() -> None:
    response = TestClient(create_app()).post("/analysis/execute", json=payload())
    assert response.status_code == 200
    body = response.json()
    assert body["trace"]["outcome"] == "partial"
    assert {result["status"] for result in body["results"]} == {"unavailable"}
    assert all(result["answer"] is None for result in body["results"])


def test_change_plan_rejects_a_single_before_image() -> None:
    response = TestClient(create_app()).post("/analysis/plan", json=payload(ImageRole.BEFORE))
    assert response.status_code == 422
