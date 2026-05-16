# Lumpyspace PINN: Inhomogeneous Cosmology via Machine Learning

Lumpyspace is a high-performance **Physics-Informed Neural Network (PINN)**
framework designed to solve for 4D inhomogeneous spacetime metrics. By utilizing
JAX-based automatic differentiation and differentiable ODE integration, the
project maps the bridge between the fundamental Einstein Field Equations (EFE)
and observational cosmological data.

## The Differentiable Cosmological Engine
Lumpyspace is more than a simulation: it is an engine that "discovers" the shape
of spacetime by looking at data. Unlike traditional tools that assume a smooth
universe from the start, this framework utilizes:
- **A Differentiable Spatetime Fabric:** A neural network that understands 4D
  geometry and strictly follows Lorentzian physics via signature enforcement.
- **Automated Tensor Calculus:** A JAX-powered engine that automatically
  calculates curvature (Ricci, Riemann), enabling direct evaluation of
  Einstein's equations.
- **A High-Performance Ray-Tracer:** A JIT-accelerated ODE solver capable of
  tracing light paths through millions of light-years in milliseconds, resilient
  to cosmological singularities.
- **A Stabilized Training Loop:** A learning environment protected by gradient
  clipping and metric regularization, capable of backpropagating through the
  entire cosmic history.

## Project Vision
The standard model of cosmology assumes large-scale homogeneity and isotropy
(FLRW). Lumpyspace explores the "Lumpy" universe—inhomogeneous solutions to
General Relativity that may better account for the observed expansion rate and
distance-redshift relationships without assuming a perfectly smooth background.

## Current Technical Capabilities

### 1. Differentiable Metric Architecture (`src/core/`)
- **MetricNN:** An Equinox-based neural network that outputs the 10 independent
  components of the 4D symmetric metric tensor.
- **Lorentzian Signature Enforcement:** The architecture strictly enforces a
  $(-, +, +, +)$ signature, ensuring the model always represents a physical
  spacetime.
- **Automated Tensor Calculus:** Implements full tensor algebra (Christoffel
  symbols, Riemann, Ricci, and Einstein tensors) using JAX's forward-mode
  automatic differentiation.

### 2. JIT-Accelerated Geodesic Ray-Tracer (`src/physics/`)
- **Differentiable ODE Integration:** Uses `diffrax` to integrate null geodesics
  backwards from an observer to high-redshift sources.
- **Performance Optimization:** Fully JIT-compiled integration loop that treats
  the neural metric as a dynamic PyTree, enabling fast backpropagation through
  the entire cosmic history.
- **Singularity Resilience:** Implements redshift-based termination events to
  gracefully handle numerical instabilities near cosmological singularities
  (e.g., the Big Bang).

### 3. Hybrid Training Infrastructure (`src/training/`)
- **Physics Loss:** Directly minimizes the residual of the Einstein Field
  Equations across a 4D spacetime volume.
- **Observational Loss:** Minimized against mock or real supernova distance
  modulus data ($\mu = 5 \log_{10}(d_L) + 25$).
- **Numerical Stability:** Built-in gradient clipping, metric regularization,
  and NaN-halting logic to ensure convergence in highly curved geometries.

## Project Roadmap

- [x] **Phase 1: FLRW Baseline** - Calibration against smooth expansion models.
- [x] **Phase 2: Tensor Algebra** - Verification of Ricci tensor logic against
  Schwarzschild vacuum.
- [x] **Phase 3: Geodesic Integration** - JIT-accelerated ray-tracing verified
  against EdS solutions.
- [ ] **Phase 4: Full Bianchi Training** - Ingestion of Pantheon+ Supernovae
  data and discovery of inhomogeneous solutions.
- [ ] **Phase 5: Analysis** - Extraction of effective $H_0$ and $q_0$ from the
  discovered metrics.

## Getting Started

### Prerequisites
- Python 3.10+
- JAX / Equinox
- Diffrax (ODE Solver)
- Optax (Optimization)

### Running Tests
The project maintains rigorous technical integrity through a comprehensive test suite:
```bash
PYTHONPATH=. pytest tests
```

---
*Lumpyspace: Navigating the curvature of the cosmos with machine learning.*
