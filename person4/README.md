PERSON 4 — CHANGE DETECTION + OPTICAL/SAR

WORK:
Build the specialist AI for:
1. Before/after change detection
2. Optical + SAR analysis

CHANGE DETECTION:

Input:
Image A + Image B

Output:
- Change mask
- Changed regions
- Change percentage
- Type of change
- Confidence

OPTICAL + SAR:

Input:
Sentinel-2 optical + Sentinel-1 SAR

Output:
- Optical analysis
- SAR analysis
- Combined interpretation
- Confidence/evidence

TOOLS:
- Open-CD
- PyTorch
- Sentinel-1
- Sentinel-2
- BigEarthNet
- Change-detection datasets

FOLDERS:

change_detection/
All before/after analysis.

optical_sar/
Optical + SAR analysis/fusion.

models/
Models used for these tasks.

tests/
Test change detection and optical/SAR results.

FINAL OUTPUT:
Provide results in a format that Person 5 can connect directly to the backend and Person 1's agent can call as a tool.

DEPENDENCY VERIFICATION (2026-09-12)

Verified with the workspace Python interpreter on Windows:

- Python 3.14.3
- PyTorch 2.14.0+cpu (no CUDA package installed)
- Rasterio 1.5.1
- NumPy 2.5.3
- MMEngine 0.10.7

Open-CD was not installed. It is not available as a compatible PyPI distribution
for this environment, and the required MMCV 2.2.0 package has no prebuilt
Windows/Python 3.14 wheel. Building that compiled dependency from source is not
appropriate for this setup, so Person 4 remains provider-based and must report
an unavailable result until a compatible OpenMMLab environment is supplied.

The following OpenMMLab components are therefore unavailable in this environment:
MMCV, MMSegmentation, MMDetection, and Open-CD. No model weights or datasets
were downloaded. Qwen-VL is owned by Person 3 and is not used by Person 4.

IMPLEMENTED CAPABILITIES

- Local bi-temporal GeoTIFF validation, including CRS, dimensions, transform,
	bounds, band count, and finite-pixel checks.
- Spatial compatibility rejection for before/after and optical/SAR pairs.
- Deterministic spectral L1 baseline with a boolean change mask, connected
	changed regions, percentage of valid pixels changed, and provenance.
- Optical and SAR descriptive baselines with a PyTorch feature-level fusion
	architecture. Each modality has its own convolutional encoder; features are
	concatenated and projected, rather than averaged at pixel level.
- Typed `SpecialistResult` adapters, structured limitations, and CPU-only
	synthetic tests.

MODEL-DEPENDENT CAPABILITIES

- Open-CD change detection remains an explicit pluggable unavailable provider.
- Learned optical/SAR interpretation remains an explicit lazy unavailable
	provider until a compatible trained model and runtime are supplied.
- The baseline fusion network is untrained and its feature vector is not an
	inference result or a calibrated confidence estimate.

LEARNED CHANGE DETECTION

The implemented learned architecture is `SiameseChangeModel`: a shared-weight
three-level convolutional encoder processes the registered before and after
images, absolute differences are formed at each encoder level, and a compact
decoder upsamples those differences to a one-channel per-pixel change logit.
The `LearnedChangeDetector` applies sigmoid thresholding, valid-pixel masking,
connected-component region extraction, and CPU inference through PyTorch.

Execution flow:

1. Rasterio reads both GeoTIFFs and validates CRS, transform, bounds, shape,
	band count, and finite pixels.
2. Each date is robustly scaled to [0, 1] and passed to the shared Siamese
	network as NCHW tensors.
3. The model returns a dense logit map; sigmoid probabilities produce the
	binary mask, percentage, regions, confidence estimate, and provenance.
4. `ChangeDetectionProvider` selects learned inference when a valid local
	checkpoint or explicit `allow_untrained=True` configuration is available.
	Otherwise it falls back to `DeterministicBaselineChangeDetector`.

CHECKPOINT STATUS

No trained or pretrained remote-sensing checkpoint is included or claimed.
The synthetic tests execute the architecture with random CPU weights only to
verify the inference path. A dataset-trained checkpoint, validation metrics,
calibration, and domain-specific labels are still required for an honest
SIH-ready learned result. Open-CD remains a pluggable unavailable provider in
this Python 3.14 Windows environment and is never simulated.

PHASE 2 REAL CHECKPOINT

Person 4 now includes the public BIT LEVIR-CD checkpoint:

- Model: BIT `base_transformer_pos_s4_dd8_dedim8`
- Source code: https://github.com/justchenhao/BIT_CD
- Checkpoint source: the Google Drive `best_ckpt.pt` linked by that repository
- Local file: `person4/checkpoints/bit_levir_best_ckpt.pt`
- Exact downloaded size: 60,048,699 bytes (about 57.3 MiB)
- Checkpoint format: PyTorch dictionary with `model_G_state_dict` and optimizer metadata
- Training dataset: LEVIR-CD building-change benchmark
- Expected input: registered RGB pair, resized to 256x256, each image normalized
	from [0, 1] to [-1, 1] using mean/std 0.5
- Runtime: standalone PyTorch adapter; CPU-compatible with the verified environment

Use it explicitly with `BITLevirChangeDetector(BITLevirConfig(checkpoint_path=...))`.
The adapter rejects non-RGB inputs, loads the state dict strictly, performs
GeoTIFF validation and resizing, and restores the output mask to the input grid.
Missing or incompatible checkpoints use the deterministic baseline through
`ChangeDetectionProvider`; they are marked `fallback-generated` in provenance.

The BIT repository states that its code is for non-commercial/research use.
That restriction and the LEVIR-CD dataset/model terms must be reviewed before
public or commercial deployment. The checkpoint is not a Sentinel-1/Sentinel-2
model: meaningful Sentinel inference still requires domain validation or a
checkpoint trained for the target sensor and task. The downloaded checkpoint
has been run successfully on a synthetic GeoTIFF CPU smoke test, not on a real
satellite scene in this workspace.

FINAL CAPABILITY MATRIX

| Capability | Status | Actual implementation | Training scope |
| --- | --- | --- | --- |
| Bi-temporal change detection | VALIDATED | BIT LEVIR adapter plus deterministic baseline fallback | BIT trained on LEVIR-CD RGB building changes |
| Sentinel-2 compatibility | IMPLEMENTED / COMPATIBILITY-TESTED | GeoTIFF validation, first-three-band RGB contract, robust normalization, BIT preprocessing | No Sentinel-2-trained checkpoint |
| Sentinel-1 compatibility | IMPLEMENTED / COMPATIBILITY-TESTED | Single-band SAR GeoTIFF validation and percentile normalization; SAR never enters BIT | No SAR-trained checkpoint |
| Optical-SAR analysis | IMPLEMENTED | Independent summaries, valid-pixel quality, registration checks, evidence regions | No semantic training claim |
| Optical-SAR fusion | ARCHITECTURE ONLY | Separate PyTorch encoders, projected fused vector, spatial fusion head | Untrained CPU architecture |
| Remote-sensing-trained checkpoint | TRAINED / DOMAIN-LIMITED | Public BIT `best_ckpt.pt`, strict 249-key load | LEVIR-CD only; not Sentinel-1/Sentinel-2 |
| Confidence | IMPLEMENTED / NOT CALIBRATED | Softmax/quality-derived confidence with explicit method | Requires target-domain calibration |
| Geographic evidence | IMPLEMENTED | Pixel-space region boxes and preserved raster metadata | Vector reprojection is not included |
| Fallback behavior | VALIDATED | Missing/incompatible learned model uses deterministic baseline or explicit failure | No fabricated learned output |

SENTINEL SAMPLE WORKFLOW

`person4/sample_data.py` records small public catalogue sources for AWS
Sentinel-2 L2A and Sentinel-1 GRD. It deliberately does not auto-download
imagery or credentials. Retrieve a small registered pair, export each to
GeoTIFF, then call `validate_downloaded_pair(optical, sar)` before using
`OpticalSarProvider`. This repository validates sensor contracts and synthetic
equivalents; it does not claim Sentinel-domain accuracy because no labelled
Sentinel pair is bundled or downloaded.

IMPLEMENTED VS VALIDATED VS TRAINED

- IMPLEMENTED: Sentinel preprocessing, registration checks, optical/SAR
	feature fusion, deterministic summaries, evidence regions, and provider
	contracts.
- VALIDATED: CPU fusion execution, synthetic Sentinel-shaped GeoTIFF tests,
	public LEVIR sample BIT inference, strict checkpoint loading, and fallback
	failure paths.
- TRAINED: BIT weights only, trained on LEVIR-CD RGB imagery.
- ARCHITECTURE ONLY: optical/SAR neural fusion and its spatial task head.
- FALLBACK: deterministic change detection and explicit unavailable learned
	optical/SAR provider.
