from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import rasterio  # type: ignore[import-untyped]
from rasterio.transform import from_origin  # type: ignore[import-untyped]

from person4.change_detection import (
    BITLevirChangeDetector,
    BITLevirConfig,
    ChangeDetectionProvider,
    DeterministicBaselineChangeDetector,
    LazyLearnedChangeProvider,
    LearnedChangeConfig,
    LearnedChangeDetector,
    SpatialCompatibilityError,
)
from person4.models import SiameseChangeModel
from person4.optical_sar import (
    BaselineOpticalSarAnalyzer,
    FeatureLevelFusion,
    LazyLearnedOpticalSarProvider,
    OpticalSarProvider,
    normalize_sentinel1_sar,
    normalize_sentinel2_rgb,
)
from person4.sample_data import SENTINEL_SAMPLE_SOURCES, validate_downloaded_pair
from shared.contracts import SpecialistStatus


def write_raster(path: Path, data: np.ndarray, transform=from_origin(100, 200, 10, 10)) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
        count=data.shape[0], dtype="float32", crs="EPSG:32632", transform=transform,
    ) as dataset:
        dataset.write(data.astype("float32"))


def test_change_baseline_returns_mask_regions_percentage_and_contract(tmp_path: Path) -> None:
    before = np.zeros((1, 4, 4), dtype=np.float32)
    after = before.copy()
    after[:, 1:3, 1:3] = 10
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    write_raster(before_path, before)
    write_raster(after_path, after)

    result = DeterministicBaselineChangeDetector(threshold=0.5).detect(before_path, after_path)

    assert result.changed
    assert result.change_mask.sum() == 4
    assert result.change_percentage == pytest.approx(25.0)
    assert result.changed_regions[0].as_dict()["pixel_count"] == 4
    contract = result.to_specialist_result(uuid4(), "change")
    assert contract.status is SpecialistStatus.COMPLETED
    assert contract.provenance["algorithm"] == "spectral_l1"


def test_change_baseline_rejects_incompatible_grids(tmp_path: Path) -> None:
    data = np.ones((1, 3, 3), dtype=np.float32)
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    write_raster(before_path, data)
    write_raster(after_path, data, from_origin(100, 200, 20, 20))

    with pytest.raises(SpatialCompatibilityError):
        DeterministicBaselineChangeDetector().detect(before_path, after_path)


def test_optical_sar_uses_feature_level_fusion_and_validates_inputs(tmp_path: Path) -> None:
    optical_path, sar_path = tmp_path / "optical.tif", tmp_path / "sar.tif"
    write_raster(optical_path, np.ones((3, 4, 4), dtype=np.float32))
    write_raster(sar_path, np.ones((2, 4, 4), dtype=np.float32) * 2)

    result = BaselineOpticalSarAnalyzer(feature_channels=8).analyze(optical_path, sar_path)
    assert result.fused_feature_shape == (1, 8)
    model = FeatureLevelFusion(3, 2, 8)
    assert model(
        np_to_tensor(np.ones((1, 3, 4, 4))), np_to_tensor(np.ones((1, 2, 4, 4)))
    ).shape == (1, 8)
    assert "untrained" in result.limitations[0]
    assert result.fused_spatial_shape == (1, 1, 4, 4)
    assert result.quality["registered"] is True
    assert result.provenance["model_output"] == "architecture-only"


def test_sentinel_preprocessing_and_sample_workflow(tmp_path: Path) -> None:
    optical = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
    sar = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    optical_norm, optical_valid = normalize_sentinel2_rgb(optical)
    sar_norm, sar_valid = normalize_sentinel1_sar(sar)
    assert optical_norm.shape == (3, 4, 4)
    assert sar_norm.shape == (1, 4, 4)
    assert optical_valid.all() and sar_valid.all()
    assert float(optical_norm.min()) == 0.0
    assert float(sar_norm.max()) == 1.0
    assert len(SENTINEL_SAMPLE_SOURCES) == 2
    optical_path, sar_path = tmp_path / "s2.tif", tmp_path / "s1.tif"
    write_raster(optical_path, optical)
    write_raster(sar_path, sar)
    manifest = validate_downloaded_pair(optical_path, sar_path)
    assert manifest["registered"] is True


def test_optical_sar_provider_returns_structured_result_and_failure(tmp_path: Path) -> None:
    optical_path, sar_path = tmp_path / "s2.tif", tmp_path / "s1.tif"
    write_raster(optical_path, np.ones((3, 4, 4), dtype=np.float32))
    write_raster(sar_path, np.ones((1, 4, 4), dtype=np.float32))
    provider = OpticalSarProvider()
    result = provider.run(uuid4(), "optical_sar", optical_path, sar_path)
    assert result.status is SpecialistStatus.COMPLETED
    assert result.provenance["model_output"] == "architecture-only"
    assert result.confidence_method == "untrained_fusion_quality_not_calibrated"
    failed = provider.run(uuid4(), "optical_sar", tmp_path / "missing.tif", sar_path)
    assert failed.status is SpecialistStatus.FAILED
    assert failed.error_code == "optical_sar_failed"


def np_to_tensor(value: np.ndarray):
    import torch

    return torch.from_numpy(value.astype("float32"))


def test_learned_providers_are_lazy_and_unavailable() -> None:
    calls: list[str] = []

    def factory():
        calls.append("loaded")
        import torch

        return torch.nn.Identity()

    change = LazyLearnedChangeProvider(factory)
    fusion = LazyLearnedOpticalSarProvider(factory)
    assert not change.model_loaded and not fusion.model_loaded and not calls
    assert change.run(uuid4(), "change").status is SpecialistStatus.UNAVAILABLE
    assert fusion.run(uuid4(), "fusion").status is SpecialistStatus.UNAVAILABLE
    assert calls == ["loaded", "loaded"]


def test_siamese_model_constructs_and_runs_on_cpu() -> None:
    import torch

    model = SiameseChangeModel(input_channels=2, base_channels=4).eval()
    before = torch.zeros((1, 2, 9, 11))
    after = torch.ones((1, 2, 9, 11))
    with torch.no_grad():
        logits = model(before, after)
    assert logits.shape == (1, 1, 9, 11)
    assert logits.device.type == "cpu"


def test_learned_detector_infers_mask_regions_and_specialist_result(tmp_path: Path) -> None:
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    before = np.zeros((1, 8, 8), dtype=np.float32)
    after = before.copy()
    after[:, 2:6, 3:7] = 4
    write_raster(before_path, before)
    write_raster(after_path, after)
    detector = LearnedChangeDetector(
        LearnedChangeConfig(input_channels=1, base_channels=4, allow_untrained=True)
    )

    result = detector.detect(before_path, after_path)

    assert result.change_mask.shape == (8, 8)
    assert result.valid_mask.shape == (8, 8)
    assert result.provenance["architecture"] == "shared_encoder_difference_decoder"
    assert "randomly initialized" in result.limitations[0]
    contract = result.to_specialist_result(uuid4(), "learned_change")
    assert contract.status is SpecialistStatus.COMPLETED
    assert contract.confidence_method == "learned_model_max_probability_not_calibrated"


def test_provider_falls_back_when_learned_checkpoint_is_unavailable(tmp_path: Path) -> None:
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    data = np.zeros((1, 4, 4), dtype=np.float32)
    write_raster(before_path, data)
    write_raster(after_path, data)
    learned = LearnedChangeDetector(
        LearnedChangeConfig(input_channels=1, checkpoint_path=tmp_path / "missing.pt")
    )

    result = ChangeDetectionProvider(learned=learned).detect(before_path, after_path)

    assert result.provenance["provider"] == "person4.deterministic_baseline"
    assert result.provenance["fallback_reason"] == "learned_model_unavailable"
    assert "baseline used" in result.limitations[-1]


def test_provider_returns_failed_specialist_result_for_bad_input() -> None:
    result = ChangeDetectionProvider().run(uuid4(), "change", "missing.tif", "after.tif")

    assert result.status is SpecialistStatus.FAILED
    assert result.error_code == "change_detection_failed"


def test_real_bit_checkpoint_loads_and_runs_on_cpu(tmp_path: Path) -> None:
    checkpoint = Path(__file__).parents[1] / "checkpoints" / "bit_levir_best_ckpt.pt"
    if not checkpoint.is_file():
        pytest.skip("downloaded BIT checkpoint is not present")
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    before = np.zeros((3, 32, 32), dtype=np.float32)
    after = before.copy()
    after[:, 10:20, 12:22] = 1
    write_raster(before_path, before)
    write_raster(after_path, after)

    detector = BITLevirChangeDetector(BITLevirConfig(checkpoint_path=checkpoint))
    result = detector.detect(before_path, after_path)

    assert detector.model_loaded
    assert result.change_mask.shape == (32, 32)
    assert result.provenance["architecture"] == "BIT base_transformer_pos_s4_dd8_dedim8"
    assert result.provenance["training_dataset"] == "LEVIR-CD"
    assert result.provenance["model_output"] == "model-generated"
    assert result.confidence_method == "bit_softmax_max_probability_not_calibrated"