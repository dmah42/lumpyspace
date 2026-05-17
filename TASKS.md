# Project Progress: Lumpyspace PINN

## Phase 1: FLRW Control Baseline (Verification & Calibration)
- [x] Task 1.1: Environment & Project Scaffolding
- [x] Task 1.2: Mock Data Generation (`data/mock_flrw.csv`)
- [x] Task 1.3: Training the Control Model
  - [x] Implement `MetricNN` architecture (`src/core/metric.py`)
  - [x] Verify signature enforcement via unit tests
  - [x] Implement automated Tensor Engine (Prerequisite Phase 2)
  - [x] Implement FLRW-constrained loss functions
  - [x] Execute calibration training loop (Sanity Check Passed)

## Phase 2: 4D Tensor Calculus Engine (The Physics Core)
- [x] Task 2.1: The Metric Wrapper
- [x] Task 2.2: Automated Tensor Algebra (`src/core/tensors.py`)
- [x] Task 2.3: Verification (Schwarzschild Vacuum Test)

## Phase 3: Differentiable Geodesic Ray-Tracer (Observational Link)
- [x] Task 3.1: The Null Geodesic ODE (`src/physics/geodesics.py`)
- [x] Task 3.2: Differentiable Integration (Diffrax implementation)
- [x] Task 3.3: Verification (FLRW Distance Match)

## Phase 4: Full Bianchi Training & Analysis
- [x] Task 4.1: Data Ingestion (Pantheon+ Supernovae)
  - [x] Implement `load_pantheon_plus` with error handling
  - [x] Implement `normalize_coordinates` for t in [-1, 1]
  - [x] Create `scripts/download_pantheon.py` for reproducible ingestion
  - [x] Verify integrity via integration tests
- [x] Task 4.2: Global Loss Optimizer
  - [x] Implement weighted chi-squared loss for observational data
  - [x] Integrate coordinate normalization into training loop
  - [x] Add full type annotations to training/physics pipeline
  - [x] Align coordinate systems (z=0 -> t=1) across data/ray-tracer
- [ ] Task 4.3: Deep Metric Extraction & Shear Analysis
- [ ] Task 4.4: Out-of-Sample Validation (Cosmic Chronometers)
- [ ] Task 4.5: Model Selection & Final Synthesis
