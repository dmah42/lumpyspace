# Comprehensive Technical Design: Inhomogeneous Cosmology PINN

## 1. Executive Summary

This document defines the architecture, physical foundation, and implementation
strategy for a Physics-Informed Neural Network (PINN) designed to reconstruct
the spacetime metric $g_{\mu\nu}$ of the universe.

**The Goal:** To model a universe that relaxes the strict homogeneity and
isotropy assumptions of the FLRW metric, using **Bianchi Class I** models as a
baseline. We aim to determine if anisotropic expansion and local backreaction
can resolve the Hubble Tension and account for Dark Energy effects without a
cosmological constant ($\Lambda$).

**The Approach:** A JAX-based PINN acts as a universal function approximator for
$g_{\mu\nu}(t, x, y, z)$, constrained by the Einstein Field Equations (EFE) and
grounded in Supernovae distance data.

---

## 2. Theoretical Foundation (The "Why")

### 2.1 The Breakdown of FLRW

The Friedmann-Lemaître-Robertson-Walker (FLRW) metric assumes a "Mean Field"
universe. By averaging out structure before solving Einstein's equations, we
ignore the non-linear "backreaction" terms ($\mathcal{Q}_D$). In General
Relativity, **averaging and evolution do not commute**:
$G_{\mu\nu}(\langle g \rangle) \neq \langle G_{\mu\nu}(g) \rangle$.

### 2.2 The Target: Bianchi Type I & Shear ($\sigma$)

We transition to **Bianchi Type I**, the simplest anisotropic generalization of
FLRW. It allows for a "Hubble Tensor" where expansion rates $H_x, H_y, H_z$ can
differ. This introduces the **shear tensor** $\sigma_{\mu\nu}$, which acts as an
effective energy density in the Raychaudhuri equation:
$$\dot{\theta} + \frac{1}{3}\theta^2 + \sigma^2 - \omega^2 = -4\pi G (\rho + 3p)$$
Our model will test if a non-zero $\sigma^2$ at late times can mimic the
acceleration attributed to $\Lambda$.

---

## 3. Implementation Stack & Structure

### 3.1 Software Stack

...

### 3.2 Engineering Standards & Style Guide

To ensure maintainability and consistency, the following standards are enforced:

- **Line Length:** Strict 80-character maximum.
- **Indentation:** 2-space indentation (no tabs).
- **Code Quality:** Enforced via `ruff` and `pre-commit`.
- **Testing:** 100% coverage required for core tensor math.
- **Type Hinting:** Mandatory for all functions (JAX `Array` types specified).

### 3.3 Directory Architecture

...

```text
lumpyspace/
├── docs/               # Technical Design & References
├── src/
│   ├── core/           # Metric wrapper, Tensor calculus (Riemann, Ricci)
│   ├── physics/        # EFE residuals, Geodesic equations, Friedmann
│   ├── models/         # Equinox-based MLP & SiREN architectures
│   ├── training/       # Loss functions, Trainers, Data loaders
│   └── utils/          # Normalization, Coordinate transforms
├── tests/
│   ├── unit/           # Math verification (e.g., Schwarzschild test)
│   └── integration/    # Training loop sanity checks
└── pyproject.toml
```

---

## 4. Mathematical Implementation of the PINN

### 4.1 Metric Topology & Constraints

The network $f_\theta: (t, x, y, z) \to \mathbb{R}^{10}$ outputs the unique
components of the symmetric metric $g_{\mu\nu}$.

- **Signature Enforcement:** We ensure a Lorentzian signature $(-, +, +, +)$ by:
  - $g_{00} = - \exp(\text{NN}_0)$
  - $g_{ii} = \exp(\text{NN}_i)$
- **Activation Function:** `SiREN` (Sine activations). Since EFE requires $C^2$
  continuity, Sine functions ensure that high-order gradients do not vanish or
  become discontinuous.

### 4.2 The Multi-Probe Loss Function

$$\mathcal{L}_{Total} = w_P \mathcal{L}_{Physics} + w_D \mathcal{L}_{Data} + w_{Reg} \mathcal{L}_{Reg}$$

1.  **Physics Loss ($\mathcal{L}_{Physics}$):** Computes the residual of
    $G_{\mu\nu} - 8\pi G T_{\mu\nu} = 0$.
    - We use `jax.jacfwd(jax.jacrev(metric))` to get the second derivatives for
      the Ricci scalar.
    - $T_{\mu\nu}$ is modeled as a perfect fluid
      $T_{\mu\nu} = (\rho+p)u_\mu u_\nu + p g_{\mu\nu}$.
    - **Expansion Prior (Symmetry Breaking):** To prevent the unconstrained
      network from falling into the mathematically valid "contracting universe"
      local minimum (like a Kasner singularity), we penalize contraction on
      _each_ individual axis: `jnp.sum(jnp.maximum(0, -H_dir)**2)`. Once the
      network enters a strictly expanding regime ($H_{x,y,z} > 0$), this penalty
      drops strictly to zero.
2.  **Data Loss ($\mathcal{L}_{Data}$):** Requires integrating the geodesic
    equation for light:
    $\frac{d^2 x^\mu}{d\lambda^2} + \Gamma^\mu_{\alpha\beta} \frac{dx^\alpha}{d\lambda} \frac{dx^\beta}{d\lambda} = 0$.
    - We solve this using `Diffrax` to find the luminosity distance $d_L(z)$ and
      compare against the **Pantheon+** dataset.
3.  **Regularization ($\mathcal{L}_{Reg}$):** Enforces the **Cosmological
    Principle** as a high-redshift prior ($z > 10$), penalizing $\sigma > 0$ in
    the early universe to remain consistent with CMB observations.

### 4.3 The Early-Universe Isotropy and Expansion Prior (The CMB Constraint)

**The Tension:** This project aims to "relax strict homogeneity and isotropy
assumptions" to test if local structure can mimic Dark Energy. However, without
observational data in the deep past, the neural metric tends to drift into
unphysical regimes (e.g., contracting pasts with high shear) because the matter
density is too low to dynamically drive expansion.

**The Resolution (The Sachs-Wolfe Effect):** We must enforce physical
constraints in the early universe ($t \le -3.0$, corresponding to $z \ge 10$)
that are consistent with Cosmic Microwave Background (CMB) observations at
recombination ($z \approx 1100$). The CMB temperature map is astonishingly
uniform, with fluctuations on the order of $\Delta T/T \approx 10^{-5}$.
According to the Sachs-Wolfe effect on large scales
($\Delta T/T \approx \frac{1}{3} \Phi/c^2$), these temperature fluctuations are
directly proportional to the primordial gravitational potential $\Phi$. This
proves with mathematical certainty that at $z \approx 1100$, the universe had
almost no "lumps" of density or massive spatial curvature gradients.
Furthermore, any significant anisotropic expansion (shear) would induce massive
quadrupole temperature fluctuations.

Because shear mathematically grows as $a^{-6}$ when integrating backwards in
time, any non-zero shear today must be vanishingly small in the early universe,
or it would explode and blatantly violate CMB observations. Furthermore, because
our model broke Bianchi Type I symmetry by developing a spatial curvature
dipole, shear is a 4D field $\sigma^2(t, x, y, z)$.

**The Implementation (Phase 4 Spec):** To ground the metric, we enforce a
localized prior as a strict boundary condition in the deep past
($t \in [-4.0, -3.990]$). Because this boundary slice is incredibly narrow, we
will explicitly sample it using a smaller batch of `N_cmb = 50` points per step.

At each sampled coordinate in this narrow slice, we must compute the Extrinsic
Curvature ($K_{ij}$) and the spatial metric ($g_{ij}$):

1. Extract the $3 \times 3$ spatial metric: $g_{ij} = g[1:4, 1:4]$.
2. Compute the spatial inverse: $g^{ij} = \text{jnp.linalg.inv}(g_{ij})$.
3. Extract the lapse function: $N = \sqrt{-g_{00}}$.
4. Compute the time derivative of the spatial metric:
   $\dot{g}_{ij} = \partial_t g_{ij}$ (extracted from `jax.jacfwd(metric)`).
5. Calculate the Extrinsic Curvature: $K_{ij} = -\frac{1}{2N} \dot{g}_{ij}$.

With these geometric primitives, we enforce three Augmented Lagrangian
penalties. Each measures its condition as a dimensionless fraction, the way
the CMB measures it, and is a linear hinge at the CMB's $10^{-5}$ accuracy
(Task 4.14). $H_{floor} = 10^{-4}$ treats smaller rates as zero.

1. **The Expansion Prior (Directional Expansion)**
   - **Math**: The directional expansion rates are the diagonal elements of the
     mixed extrinsic curvature tensor. $H_i = -K^i_i$ (no sum over $i$).
   - **Penalty**: the share of the squared directional rates that is
     contraction,
     $f_e = \sum_i \max(0, -H_i)^2 / (\sum_i H_i^2 + H_{floor}^2) \in [0, 1]$,
     and $\mathcal{L}_{\text{expand}} = \max(0, f_e - 10^{-10})$, which
     penalizes contraction on _any_ individual axis to prevent "Kasner-like"
     local minima.

2. **The Isotropy Prior (Shear $\sigma^2$)**
   - **Math**: The traceless shear tensor is
     $\sigma_{ij} = K_{ij} - \frac{1}{3} \text{Tr}(K) g_{ij}$.
   - **Scalar Formulation**: The shear scalar is
     $\sigma^2 = \frac{1}{2} \sigma_{ij} \sigma^{ij} = \frac{1}{2} (g^{ia} g^{jb} \sigma_{ij} \sigma_{ab})$.
   - **Penalty**: the shear fraction
     $f_\sigma = \sigma^2 / (H^2 + \sigma^2 + H_{floor}^2) \in [0, 1)$, which is
     $(\sigma/H)^2$ while the shear is small, and
     $\mathcal{L}_{\text{shear}} = \max(0, f_\sigma - 10^{-10})$.

3. **Spatial Homogeneity (Curvature Gradients)**
   - **Math**: We penalize the spatial gradients of all metric components. Let
     $\partial_k g_{\mu\nu}$ be the spatial Jacobian (derivatives with respect
     to $x, y, z$).
   - **Penalty**: the perturbation amplitude
     $A = \sqrt{\sum_{k=1}^3 \sum_{\mu,\nu=0}^3 (\partial_k g_{\mu\nu})^2}$ per
     unit code length, and
     $\mathcal{L}_{\text{spatial}} = \max(0, A - 10^{-5})$.

**Augmented Lagrangian Architecture:** Because these three penalties represent
distinct physical constraints, summing them under a single $\lambda$ multiplier
would cause gradient imbalance (e.g., the network trades shear violation for
expansion violation).

- **Independent Tracking**: The model state must be updated to track three
  separate `AdaptivePenaltyState` schedulers and three independent $\lambda$
  multipliers (`lambda_cmb_expand`, `lambda_cmb_shear`, `lambda_cmb_spatial`).
- **Initialization**: Initialize all three penalties with a starting weight
  $w = 100$ and $\lambda = 0$ (section 5.2.4).

**Stitching the Boundary (EFE Overlap):** Applying these penalties at
$t \in [-4.0, -3.990]$ acts as a discrete boundary snapshot. To connect this
boundary to the late universe, the Einstein Field Equations (EFE) must be
stitched seamlessly to it. Therefore, the continuous EFE sampling domain
(`t_inactive`) must be extended down to $t=-3.990$. Because the EFE physical
domain touches the CMB boundary, the differential equations will "grab" the
expanding, isotropic boundary condition and mathematically propagate it forward
through the unobserved region all the way into the active Supernova region.

_Note on Stiffness_: Extending the EFE sampling down into the $-3.990$ region
will introduce incredibly massive density gradients ($\rho(z) \propto (1+z)^3$).
Testing the EFE extension (moving the `minval` from $-3.9$ to $-3.990$) to
ensure the Spatial Curriculum can handle the mathematical stiffness without
`NaN` explosions is a critical prerequisite before activating the CMB boundary
loss.

---

## 5. Phased Implementation Roadmap

### Phase 1: The FLRW Control Baseline (Verification & Calibration)

**Goal:** Prove the 4D PINN architecture can perfectly recover the standard
$\Lambda$CDM model. This serves as our "Control Group" to ensure the tensor
engine and loss functions are bug-free.

- **Task 1.1: Environment & Project Scaffolding**
  - Initialize `pyproject.toml` with the JAX/Equinox stack.
  - Set up the directory structure as defined in Section 3.2.
- **Task 1.2: Mock Data Generation**
  - Create a script `scripts/generate_mock_flrw.py` to produce a "Perfect FLRW"
    universe.
  - Generate a dataset of $d_L(z)$ values for $z \in [0, 2]$ using standard
    $\Lambda$CDM parameters ($\Omega_m=0.3, \Omega_\Lambda=0.7, H_0=70$).
- **Task 1.3: Training the Control Model**
  - Initialize the 4D Metric PINN but enforce **Isotropy Constraints**
    ($g_{xx} = g_{yy} = g_{zz}$ and all off-diagonals = 0).
  - Physics Loss: Include a non-zero $\Lambda$ term to match the mock data:
    $\mathcal{L}_{phys} = \|G_{\mu\nu} + \Lambda g_{\mu\nu} - 8\pi G T_{\mu\nu}\|^2$.
- **Task 1.4: Validation & Verification (V&V)**
  - **Scale Factor Recovery:** Extract $a(t) = \sqrt{g_{xx}(t, 0)}$ from the
    trained network. It must match the analytic FLRW scale factor within 0.1%.
  - **Residual Check:** After training, the $G_{\mu\nu}$ components must satisfy
    the 1st and 2nd Friedmann equations exactly.
  - **Metric Signature:** Verify via `jax.numpy.linalg.eigvals` that the
    signature remains $(-, +, +, +)$ at all points in the training domain.

### Phase 2: The 4D Tensor Calculus Engine (The Physics Core)

... (renumbered from here) ...

**Goal:** Construct a robust library to compute $G_{\mu\nu}$ from any network
$g_{\mu\nu}(t, x, y, z)$.

- **Task 2.1: The Metric Wrapper (`src/core/metric.py`)**
  - Implement `MetricTensor(equinox.Module)`:
    - Input: $(t, x, y, z)$. Output: 10 scalars representing the upper triangle
      of $g_{\mu\nu}$.
    - Hard-code the signature: $g_{00} = -\exp(out_0)$, $g_{ii} = \exp(out_i)$,
      off-diagonals = $out_j$.
- **Task 2.2: Automated Tensor Algebra (`src/core/tensors.py`)**
  - **Step A: Jacobian of the Metric.** Use `jax.jacfwd` to get
    $\partial_\gamma g_{\mu\nu}$.
  - **Step B: Inverse Metric.** Use `jax.lax.linalg.inv` (crucial for
    Christoffel symbols).
  - **Step C: Christoffel Symbols.** Implement
    $\Gamma^\alpha_{\mu\nu} = \frac{1}{2} g^{\alpha\sigma}(\partial_\mu g_{\nu\sigma} + \partial_\nu g_{\mu\sigma} - \partial_\sigma g_{\mu\nu})$.
  - **Step D: Ricci Tensor.** Implement
    $R_{\mu\nu} = \partial_\rho \Gamma^\rho_{\mu\nu} - \partial_\nu \Gamma^\rho_{\mu\rho} + \Gamma^\rho_{\mu\nu}\Gamma^\sigma_{\rho\sigma} - \Gamma^\sigma_{\mu\rho}\Gamma^\rho_{\nu\sigma}$.
    - _Note:_ This requires the Hessian of the metric. Use
      `jax.jacfwd(jax.jacrev(model))` for maximum stability.
- **Task 2.3: Verification (The Schwarzschild Vacuum Test)**
  - Create `tests/unit/test_tensors.py`.
  - Define an analytic function for the Schwarzschild metric:
    $g_{00} = -(1-2M/r)$, $g_{rr} = (1-2M/r)^{-1}$, etc.
  - Pass this function into the tensor engine.
  - **Success Criterion:** All components of $R_{\mu\nu}$ must be zero to within
    $10^{-5}$ tolerance.

### Phase 3: Differentiable Geodesic Ray-Tracer (The Observational Link)

**Goal:** Map the learned metric to the Luminosity Distance $d_L(z)$ using
`Diffrax`.

- **Task 3.1: The Null Geodesic ODE (`src/physics/geodesics.py`)**
  - Define the system of first-order ODEs:
    1.  $\frac{d x^\mu}{d\lambda} = k^\mu$
    2.  $\frac{d k^\mu}{d\lambda} = -\Gamma^\mu_{\alpha\beta} k^\alpha k^\beta$
  - Constraint: Light follows null paths ($g_{\mu\nu} k^\mu k^\beta = 0$).
- **Task 3.2: Differentiable Integration**
  - Use `diffrax.diffeqsolve` with `Tsit5()` (Runge-Kutta 5/4).
  - Implement the observer-to-source integration: Start at $t=now, \vec{x}=0$,
    integrate backwards in affine parameter $\lambda$ until the desired redshift
    $z$ is reached.
  - Calculate $d_L(z) = (1+z) r_{prop}$.
- **Task 3.3: Verification (FLRW Distance Match)**
  - Set the metric network to a frozen FLRW state.
  - Run the ray-tracer to compute $d_L(z)$ for $z \in [0, 1.5]$.
  - **Success Criterion:** $d_L(z)$ must match the analytic FLRW integral
    $\int dz/H(z)$ within 1%.

### Phase 4: Full Bianchi Training & Analysis (The Discovery Phase)

**Goal:** Train the full 4D PINN on the Pantheon+ Supernova dataset and analyze
for shear.

- **Task 4.1: Data Ingestion (`src/training/data.py`) [COMPLETED]**
  - Load `Pantheon+` Supernovae: $(z_{cmb}, \mu_{obs}, \sigma_\mu)$.
  - Implement coordinate normalization: Map $z \in [0, 2]$ to $t \in [-1, 1]$.
- **Task 4.2: The Global Loss Optimizer [COMPLETED]**
  - **Learning Rate Scheduler:** Implement an
    `optax.warmup_cosine_decay_schedule` to navigate the rugged Tabula Rasa loss
    landscape. The schedule must be parameterized (`warmup_steps`,
    `decay_steps`) to provide a sharp "kick" out of local minima followed by a
    smooth descent.
  - **Optimizer:** Initialize
    `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))`.
  - **The Training Loop:**
    1.  Sample $10,000$ random points $(t,x,y,z)$ for $\mathcal{L}_{Physics}$.
    2.  Compute $G_{\mu\nu}$ at these points. Minimize $\|G_{\mu\nu}\|^2$
        (assuming vacuum/dust).
    3.  Compute $d_L(z)$ via Task 3.2 for the Supernova redshifts.
    4.  Minimize $\chi^2 = \sum (\mu_{pred} - \mu_{obs})^2 / \sigma_\mu^2$.

- **Task 4.3: Automated Hyperparameter Optimization (Gradient Balancing)
  [COMPLETED]**
  - **Goal:** Automate the tuning of the multi-probe loss weights ($w_{sn}$,
    $w_{bao}$) to prevent the physics loss ($G_{\mu\nu}=0$) and data losses from
    overwhelming each other.
  - **Implementation Options:**
    1.  **Bayesian Optimization (Optuna/Ray Tune):** Wrapping the training loop
        in an external framework to run dozens of trials, hunting for the
        optimal static weights that minimize validation loss.
        - _Tradeoffs:_ Highly robust and guaranteed to find stable static
          weights. However, it is computationally expensive (requires running
          the full PINN training loop dozens of times) and uses static weights
          that cannot adapt if the loss landscape changes drastically during
          training.
    2.  **Self-Adaptive Loss Weights (The PINN Approach):** Making the weights
        ($w_{sn}$, $w_{bao}$) learnable parameters within the JAX/Equinox model
        itself. The loss function is modified to simultaneously minimize the
        residuals while maximizing the weights (e.g.,
        $\mathcal{L} = \sum \frac{1}{2 w_i^2} \mathcal{L}_i + \log(w_i)$),
        allowing the optimizer to dynamically balance the gradients at every
        step.
        - _Tradeoffs & Failure Mode:_ While computationally elegant, this method
          suffers from the "Lazy Optimizer" problem when applied to highly rigid
          differential equations like the EFE. If the metric initialization
          produces massive curvature gradients that are difficult to minimize,
          the optimizer finds it mathematically cheaper to simply drive
          $w_i \to \infty$. This zeros out the effective loss
          $\left( \frac{1}{2 w_i^2} \to 0 \right)$, stalling training
          completely. In our empirical testing, meticulously hand-tuned static
          hyperparameter weights proved far superior as they act as unyielding
          constraints that force the network to actually solve the PDEs.

- **Task 4.4: Deep Metric Extraction & Shear Analysis [COMPLETED]**
  - **Goal:** Quantify the physical deviations from FLRW by extracting the shear
    tensor and expansion rates.
  - **Implementation:**
    - Compute the **Shear Tensor** $\sigma_{\mu\nu}$ using the first derivatives
      of the metric.
    - Calculate the scalar shear
      $\sigma^2 = \frac{1}{2}\sigma_{\mu\nu}\sigma^{\mu\nu}$.
    - Extract the **Hubble Tensor** $H_{ij}$ and compare expansion rates along
      the x, y, and z axes.
    - Generate the final **Hubble Diagram** showing the Pantheon+ fit, CC
      residuals, and shear evolution.
  - **Success Criterion:** Identification of the primary expansion axis and its
    alignment with known cosmological structures (e.g., CMB dipole).

- **Task 4.5: Sparse Observational Datasets (Cosmic Chronometers & BAO)
  [REJECTED]**
  - **Goal:** Integrate additional temporal and distance measurements to further
    constrain the metric beyond Supernova data.
  - **Implementation:**
    - _Cosmic Chronometers (CC):_ Utilizing directional $dt/dz$ measurements
      from passively evolving galaxies.
    - _Baryon Acoustic Oscillations (BAO):_ Utilizing 3D standard ruler data
      derived from early universe plasma physics.
  - **Reason for Rejection:** Both the CC and BAO datasets are currently too
    sparse (e.g., ~30 high-quality CC data points) to provide a robust training
    signal. Incorporating them would risk overfitting the highly flexible
    10-dimensional metric network to small-sample noise rather than generalized
    physical geometry.

- **Task 4.6: Advanced Probes & Geometries [COMPLETED]**
  - _CMB Temperature Power Spectrum:_ Mapping our 4D metric fluctuations to the
    CMB angular power spectrum to ensure the late-time shear does not violate
    the high-redshift isotropy of the CMB ($z \approx 1100$).
  - _Inhomogeneous and Anisotropic Spacetimes (Szekeres):_ Relaxing the Bianchi
    Type I symmetry to allow for full spatial inhomogeneity while preserving
    directional expansion. We can model Szekeres spacetimes analytically to test
    if localized curvature gradients and non-zero shear can simultaneously
    explain both the Hubble Tension and the illusion of Dark Energy.

- **Task 4.7: Dynamic Kick Period (Cosine Annealing with Warm Restarts)
  [COMPLETED]**
  - **Goal:** Smooth out late-stage training by transitioning from rapid 
    early-stage exploration to long, stable settling periods.
  - **Implementation:** Modify the `lr_schedule` in `trainer.py` to use a 
    geometric progression for the `kick_period` using $T_i = T_{i-1} \times T_{mult}$. 
    To preserve exponential settling time without causing kick starvation over a 
    30,000 step run, use $T_0 = 500$ and $T_{mult} = 1.5$. This yields 8 kicks 
    over 30,000 steps while mathematically aligning with the exponentially 
    increasing time required to navigate deep local minima.

- **Task 4.8: Late-Time Isotropy Penalty (Galaxy Surveys) [FUTURE]**
  - **Goal:** Align the model's geometry with macroscopic observations of the
    local universe today ($t=0$).
  - **Implementation:** Observational data from all-sky local galaxy surveys
    (e.g., [2MASS](https://www.ipac.caltech.edu/2mass/)) indicates that the
    macroscopic distribution of matter and expansion is isotropic to within
    roughly 1% to 2%. To match this observation, we will implement a soft
    Augmented Lagrangian penalty at $t=0$. The penalty will only trigger if the
    variance between the directional Hubble parameters ($H_x, H_y, H_z$) exceeds
    a 2% threshold. This anchors the metric to the physical reality of the local
    universe while mathematically allowing for small, physically viable
    anisotropic solutions.

- **Task 4.9: Model Selection & Final Synthesis [FUTURE]**
  - **Goal:** Statistically determine if the inhomogeneous model is superior to
    $\Lambda$CDM.
  - **Implementation:**
    - Compute the **AIC (Akaike Information Criterion)** and **BIC (Bayesian
      Information Criterion)** for both the PINN and a standard FLRW fit.
  - **Success Criterion:** Final determination of whether the data statistically
    justifies a "lumpy" universe over a smooth one.

- **Task 4.10: Inhomogeneity Seed & Frequency Ablation Suite [FUTURE]**
  - **Goal:** Rigorously verify that the discovered "Lumpyspace" geometry
    (anisotropic shear channels and localized curvature lumps) and the derived
    matter density ($\Omega_m \approx 0.23$) are genuine physical attractors
    demanded by the Pantheon+ data and Einstein Field Equations, rather than
    unearned artifacts or echoes of the spatial initialization seed.
  - **Implementation:**
    - **Spatial Frequency Sweep ($\omega_0$):** Train an ensemble of models
      across a broad range of first-layer SIREN coordinate scaling factors:
      $\omega_0 \in \{3.0, 5.0, 10.0, 15.0, 20.0, 30.0\}$. Evaluate whether
      low frequencies (box-scale variations) and high frequencies (fine-scale
      ripples) coalesce into the same macroscopic shear channels, or if the
      characteristic lump scale is bounded by the initialization frequency.
    - **Stochastic RNG Seed Ensemble:** Train replicas across varied PRNG keys
      (`seed` $\in \{42, 1337, 2024, 7, 999\}$) to ensure the orientation,
      dipole alignment, and backreaction dynamics are invariant to specific
      initial pseudo-random weight configurations.
  - **Evaluation Metrics:**
    - Stability and convergence of $\Omega_m$ across runs.
    - Cross-correlation of the resulting 3D shear field $\sigma^2(t, x, y, z)$
      and Ricci scalar curvature $R(t, x, y, z)$.
    - Supernova $\chi^2$ and distance modulus residual distributions.
    - Early-universe CMB boundary constraint adherence ($l_{\text{expand}}$,
      $l_{\text{shear}}$, $l_{\text{spatial}}$).
  - **Success Criterion:** The macroscopic geometry, expansion history, and
    Pantheon+ fit remain consistent across seeds, demonstrating that the
    lumpiness is an emergent physical necessity of inhomogeneous cosmology.

- **Task 4.11: CMB Collocation Sampling Density [FUTURE]**
  - **Goal:** `N_cmb = 50` (the CMB boundary slice's point count) is 20x
    sparser than the `N = 1000` used for the main EFE/spatial-weight
    sampling, despite both covering the same $[-1,1]^3$ spatial cube — the
    CMB block's narrow time window ($t \in [-4.0, -3.990]$) only justifies
    fewer _time_ samples, not fewer _spatial_ ones. Because
    `t_cmb`/`spatial_cmb` are combined via row-wise
    `jnp.concatenate(..., axis=1)`, the two are currently forced to share one
    count, so the narrow-slice justification silently starves spatial
    coverage too. This risks under-sampling the localized curvature/shear
    "lumps" that are the whole thesis of this project, and adds noise to the
    CMB loss terms and to the selection score (Task 4.13), whose shear,
    expansion and homogeneity terms are means over the same 50 points,
    redrawn every step.
  - **Implementation:** Either (a) raise `N_cmb` as a simple shared count
    (a moderate ~200 gives a real but modest noise reduction; matching
    `N=1000` fully matches EFE's spatial density at 20x the CMB compute), or
    (b) decouple the two axes structurally — e.g. broadcast/tile a small set
    of distinct time-locations across an independently-sized, larger spatial
    batch, instead of the current 1:1 pairing.
  - **Scope:** the CMB terms only. `l_wec` is evaluated on the `N=1000` EFE
    points, and its plateau under growing AL pressure traced to the spatial
    weighting absorbing its penalty (Task 4.14), not to sampling density.
  - **Success Criterion:** `l_spatial`'s achieved minimum and variance
    measurably change (in either direction) at higher `N_cmb`, and the
    step-to-step spread of the score's CMB terms at a fixed training state
    shrinks.

- **Task 4.12: Unreachable-Redshift Defect & Observational Channel Rebuild
  [FUTURE]**
  - **Symptom (measured 2026-09-28, checkpoint at step ~10.6k):** null
    geodesics accumulate a total redshift of $z \approx 2\times10^{-5}$ over
    the _entire_ affine path (`sol.t1` reaches `L_MAX = -10` every time).
    100% of sampled Pantheon+ supernovae are unreachable — including the
    nearest at $z = 0.0012$, still ~60x beyond range.
  - **Mechanism:** `get_luminosity_distance` does not fail on unreachable
    targets. `jnp.interp` clamps to the trajectory endpoint, and the
    "Optimizer Compass" adds `10 * (z_target - max_z_found)`. The returned
    value reduces to $d_L \approx 10^4 z - 105$ Mpc — a linear function of
    the _target_ redshift with a near-constant geometric offset. Measured:
    $z=0.0030 \to -75.3$ (formula $-75$); $z=0.0199 \to 94.1$ ($94$);
    $z=0.0283 \to 177.7$ ($178$). Negative (unphysical) distances are then
    silently absolved by `jnp.abs(dl_mpc)` in `get_data_loss`.
  - **Consequence:** `l_sn` has been measuring redshift shortfall, not fit
    quality. Measured per-SN $\chi^2$ on a random subsample: 46, matching the
    logged 29-90 range. **No distance-redshift physics has entered the
    objective.** The defect dates to `d62abd2` (2026-05-17), so results from
    blog posts 4 and 5 both require re-verification. The ray-tracer itself is
    sound — `test_flrw_distance_match` validates it to 5% against analytic
    EdS; it is being handed a static spacetime.
  - **Root cause — Minkowski is a global minimum of every physics term:**
    $\mathcal{L}_{phys}$ ($G=0$, and `kappa_rho_0 = omega_m * 3 * h_mean**2`
    zeroes the stress-energy whenever $H \to 0$, _regardless of_ $\Omega_m$),
    $l_{wec}$ ($R=0$), $l_{expand}$ ($H_i=0$), $l_{shear}$ ($\sigma^2=0$),
    $l_{spatial}$ (no gradients). Only the supernova term can push the metric
    off flat static space, and that channel was severed.
  - **Reachability-honest SN loss [COMPLETED].**
    `get_luminosity_distance` returns `(dL, z_reached)` and invents nothing for
    an unreachable target. `l_sn` is the $\chi^2$ over reachable supernovae
    with a _fixed_ denominator of the full catalogue, so a badly-fitting
    supernova cannot be shed by making it unreachable. `l_sn_cover` carries the
    mean redshift shortfall as its own term. `z_reached` is the redshift at the
    trajectory's last finite state, which denies a ray that redshifts briefly
    and then blueshifts back. Directional ray-tracing (per-supernova RA/Dec,
    with the null normalization generalized to include the $g_{0i}$ shift
    terms) landed alongside it.
  - **Outcome (measured 2026-10-01, step ~1.8k):** the channel reconnected.
    `l_sn_cover` fell from $0.25$ to $8.6\times10^{-5}$; `l_sn` fell from the
    fabricated $46$ to $0.538$. The supernova fit is now genuine. $H_{mean}$ at
    $t=1$ sits at $\approx 0.002$ code units, under 1% of the observed $H_0$,
    and $a_{mean}$ is flat across $t \in [-4, 1]$ — near-zero net volume
    expansion from $z\approx999$ to today.
  - **Consequent degeneracy:** redshift in a general metric has two sources,
    metric expansion and the gravitational potential via $\sqrt{|g_{00}|}$. The
    ray-tracer measures their sum and `l_sn` constrains their sum, so either
    channel alone satisfies it. The model currently fits Pantheon+ almost
    entirely through $\sqrt{|g_{00}|}$. This is the part-4 degeneracy in
    sharper form: the geometry now fits the data, through the wrong mechanism.
  - **Absolute density parameterization [COMPLETED].** The trained parameter
    is an absolute density $\kappa\rho_0$, bounded between the baryons and
    the total matter (section 5.4). Flat static space with $\rho_0 > 0$
    yields a nonzero EFE residual, so "there is matter here" is incompatible
    with empty static space — a real expansion gradient. $\Omega_m$ is a
    _derived, reported_ quantity, $\kappa\rho_0 / 3H_{mean}(t{=}1)^2$. This
    targets the measured failure: with $h_{mean} \approx 0.002$, a density
    derived as $\Omega_m \cdot 3h_{mean}^2$ was starved to
    $\sim 10^{-5}$, and a universe with no matter in it has no reason to
    expand.
  - **Outcome (measured 2026-10-18, best checkpoint at step 1310):** the
    matter coupling holds. The EFE and WEC selection terms fell by about
    eight orders of magnitude; the unbalanced curvature is about 20 times
    $\kappa\rho$, down from about $10^6$. $\kappa\rho_0$ reached the
    total-matter ceiling ($0.0467$) by step 130 and stayed there. $H_{mean}$
    at $t=1$ is $\approx 0.0035$, 1.4% of $H_0$, so the reported $\Omega_m$ is
    $\approx 1300$. The Friedmann constraint
    $3H^2 - \sigma^2 + \tfrac12 {}^3R = \kappa\rho$ asks for
    $H \approx 0.125$ in flat isotropic space, and the model balances its
    matter with spatial curvature. $l_{sn} = 1.00$.
  - **Present-day expansion band [FUTURE].** A two-sided hinge on
    $H_{mean}$ at the observer, $(t, x, y, z) = (1, 0, 0, 0)$, over
    $[60, 80]$ km s⁻¹ Mpc⁻¹, or $[0.200, 0.267]$ in code units. It is
    exactly zero inside the band, joins the Augmented Lagrangian set
    alongside the CMB priors, and adds a term to the selection objective
    (Task 4.13). `get_cosmology_today()` already computes the required
    quantity. Its violation measure and starting weight are specified in
    Task 4.14.
    - **What it assumes.** The SH0ES value is a local distance ladder:
      geometric anchors calibrate Cepheids, Cepheids calibrate Type Ia
      absolute magnitudes, and $H_0$ is the velocity-distance slope over
      $0.01 < z < 0.15$. Homogeneity enters only through peculiar-velocity
      corrections and local density contrast, both bounded at the 1-2% level.
      The Hubble-flow fit fixes the deceleration parameter at the
      $\Lambda$CDM value $q_0 = -0.55$; a shift of $0.5$ in $q_0$ moves $H_0$
      by about 2%, well inside the band's $\pm 14\%$.
    - **Where it applies.** For any spacetime, the distance-redshift relation
      of comoving sources near a comoving observer is
      $cz \approx d\,(\theta/3 + \sigma_{ij} n^i n^j)$ at the observer
      (Kristian & Sachs 1966). The shear term is a quadrupole on the sky, so
      the sky-averaged slope that SH0ES measures is $\theta/3$ at the
      observer's own point. The band therefore constrains that single point.
      An average over a spatial volume would identify the measured $H_0$ with
      the volume-averaged expansion rate, which differs from it in an
      inhomogeneous universe by the backreaction this project tests. The
      higher-order terms over the fitted redshift range are of order a few
      percent, inside the band.
      (The Planck $67.4$ is the model-dependent figure, inferred from the
      sound horizon under $\Lambda$CDM.) A band also carries no information
      the data lacks — the slope of $\mu(z)$ in Pantheon+ _is_ $H_0$, and the
      model is already told 1701 times. It governs which channel carries it.
    - **Why it concedes nothing under test.** This project asks whether
      inhomogeneity can account for cosmic _acceleration_ without dark energy.
      Expansion at roughly 70 km s⁻¹ Mpc⁻¹ is measured and was never the
      contested claim; $\Lambda$ is. Note the asymmetry with the $\Omega_m$
      clip that the absolute density parameterization removes: $\Omega_m$
      constrains what the universe is made of, which is under test, where $H$
      constrains how fast it expands, which is not.
  - **Anisotropic time-symmetry breaking at init [FUTURE].** Minkowski
    is maximally symmetric (10-parameter isometry group) and is therefore
    _not_ a neutral prior — it assumes perfect homogeneity, perfect isotropy
    and staticity simultaneously. SIREN $\omega_0$ already addressed the
    spatial half of this (see blog post 5); the temporal half was never
    addressed. Seed the spatial components to grow in $t$ at _different,
    randomly-drawn rates per axis_: that breaks staticity without assuming
    FLRW, since expansion rate, history, anisotropy and any convergence
    toward isotropy all remain learned. Explicitly **not** an FLRW
    background or FLRW pretraining — both would bake in the homogeneity and
    isotropy this project exists to interrogate, and would make Task 4.10's
    initialization-echo ablation unanswerable.
  - **Sequencing:** accuracy-normalised selection (Task 4.13) lands first.
    Then the absolute density parameterization goes alone,
    keeping the change minimal so its effect on $H_{mean}$ can be observed in
    isolation. It assumes strictly less than the expansion band does, so it
    gets the first attempt: if decoupling the density from the expansion rate
    is enough to lift $H$ into the observed range unaided, that is a materially
    stronger result than clamping it there. The expansion band follows only if
    $H_{mean}(t{=}1)$ is still outside it, which the outcome above shows it
    is. Time-symmetry breaking at init is warranted only if the two together
    cannot escape the static basin.
  - **Success Criterion:** `l_sn` holds a reduced $\chi^2$ consistent with a
    genuine Pantheon+ fit while `l_sn_cover` stays at zero, $H_{mean}(t{=}1)$
    lands inside $[0.200, 0.267]$, and $\Omega_m$ is reported by the fit rather
    than clipped.

- **Task 4.13: Accuracy-Normalised Best-Checkpoint Selection [COMPLETED]**
  - **Scope:** selection only. The training loss, its weights and the
    Augmented Lagrangian are unchanged. This task decides which step is saved
    as the best checkpoint, and therefore when the cycle-based stop rule
    fires, since that rule keys off a new best.
  - **Current selection and its problems:** the best checkpoint is the lowest
    `l_sn` among steps where each of `l_wec`, `l_expand`, `l_shear` and
    `l_spatial` is at most `FEASIBILITY_EPS = 0.5`.
    - Physics enters only as pass/fail. `l_phys`, the field-equation residual,
      plays no part in the choice.
    - 0.5 was calibrated from observed loss values. In physical terms it
      permits contraction rates up to 2-3x $H_0$, $\sigma^2$ up to ~0.71,
      $\sum(\partial g)^2$ up to ~0.71, and mean negative $R$ about 10x the
      real universe's $\kappa\rho_0 \approx 0.053$. Three of the four checks
      never bind.
    - Ranking by lowest `l_sn` below $\chi^2/N \approx 1$ rewards overfitting.
      The step-3282 checkpoint of the current run sits at 0.46.
  - **Selection objective:** one fixed scalar $S$, the sum of the terms
    below. Each term is a mean over its sample points of a squared deviation
    in units of that quantity's accuracy, so a term of order 1 means "as good
    as the accuracy allows", exactly as $\chi^2/N$ does for the supernovae.
    The best checkpoint is the lowest $S$, with no gate.

    | term | quantity | accuracy |
    | --- | --- | --- |
    | SN fit | `l_sn` ($\chi^2/N$, fixed denominator) | Pantheon+ $\mu$ errors, already applied |
    | SN coverage | per-SN shortfall $\max(0, z_i - z_{reached,i})$ | $\sigma_z \approx 10^{-3}$ (peculiar-velocity redshift scatter) |
    | Shear | $\sigma/H$ at the CMB boundary | $10^{-5}$ (CMB quadrupole) |
    | Expansion | $\max(0, -H_i)/H$ per axis at the CMB boundary | $10^{-5}$ (CMB) |
    | Homogeneity | metric gradient amplitude $\sqrt{\sum(\partial_k g_{\mu\nu})^2}$ over the ~1 Gpc sample scale | $10^{-5}$ (Sachs-Wolfe, $\Phi \sim 10^{-5}$) |
    | EFE | residual $\lvert G - \kappa T \rvert$ relative to the local matter term $\kappa\rho$ | $10^{-2}$ (see below) |
    | WEC | $\max(0, -R)$ relative to $\kappa\rho$ | $10^{-2}$ (see below) |

  - **EFE and WEC accuracy:** these are theoretical conditions, so their
    accuracy is chosen. $\sigma_{EFE} = \sigma_{WEC} = 10^{-2}$: the field
    equations and the energy condition must hold to 1% of the local matter
    term. Inhomogeneous alternatives to $\Lambda$ predict density effects of
    order 10%, so a residual an order of magnitude smaller keeps an effect of
    that size distinguishable from an unbalanced equation. Observational
    estimates of the matter density (Planck's $\Omega_m$, $\omega_m$) are
    inferred within $\Lambda$CDM and are not used to set it.
  - **Comoving kinematics:** $H_i$, $H = \theta/3$ and $\sigma^2$ are those of
    the matter congruence $u^\mu = \delta^\mu_0/\sqrt{-g_{00}}$, the same one
    the stress-energy tensor uses: strain rate
    $D_{ij} = \partial_t h_{ij} / (2\sqrt{-g_{00}})$ with the rest-frame
    spatial metric $h_{ij} = g_{ij} - g_{0i}g_{0j}/g_{00}$ (Landau & Lifshitz
    §84). With zero shift this reduces to $\partial_t g_{ij}/(2\sqrt{-g_{00}})$;
    with a shift it carries the terms that formula drops. One function,
    `src/core/kinematics.py`, serves the selection score, the CMB loss,
    `get_cosmology_today()` and the monitoring notebook. Its use in the CMB
    loss and in $\kappa\rho_0$ changes training as well as selection.
  - **Vanishing scales:** $H$ and $\kappa\rho$ go to zero on flat static
    space, so each ratio needs the same small-denominator guard style as
    `get_efe_loss`. With that guard, Minkowski scores zero on every physics
    term, which is correct: it is vacuum, isotropic, homogeneous and not
    contracting. Only the SN coverage term excludes it, as in the
    loss. `best_warmup_steps` keeps the near-Minkowski opening steps out of
    selection.
  - **Expected consequence:** the CMB terms dominate $S$ wherever the
    geometry violates them. The current run's boundary sits at
    $\sigma/H \approx 2$, a shear term of order $10^{10}$, so selection there
    reduces in practice to "least CMB violation" until that violation comes
    within range. That is the intended behaviour: violations count in
    proportion to how precisely they are measured.
  - **Implementation:**
    - The step function returns the per-term quantities as auxiliary
      outputs. No gradient flows through them, so optimisation is untouched.
    - $S$ is a column in `training_metrics.csv`, written every logged step,
      and the monitoring notebook plots it alongside the loss chart. $S$ and
      its per-term breakdown are printed on each new best and stored in the
      best checkpoint's `.meta`.
    - Resume reads the best $S$ and its step from the best checkpoint's
      `.meta` when it exists.
    - Lands with a fresh run, so the `.meta` schema cleanup goes with it and
      needs no migration: every field describes its own checkpoint (`step`,
      `l_sn`, $S$ and its terms), and the Augmented Lagrangian states sit
      under `constraints` keyed `wec`, `expand`, `shear`, `spatial`.
    - A resumed run starts at the step after the saved one, so no step is
      logged twice.
  - **Tests:** each term against metrics with known values (an FLRW metric
    scores zero shear and zero contraction; a Kasner metric has an analytic
    $\sigma/H$; Bell's spaceship has expansion carried entirely by the shift
    terms); selection prefers lower $S$ regardless of which terms differ;
    resume restores the best $S$ and step; a resumed run logs each step once.
  - **Sequencing:** lands before the absolute density parameterization and
    the expansion band (Task 4.12), so every run that tests those is judged
    by $S$.
  - **Success Criterion:** the best checkpoint is the step with the lowest
    $S$, and its per-term breakdown states in accuracy units how far the
    geometry sits from the supernova data and from each physical condition.
  - **Score noise [FUTURE]:** $S$ swings by two orders of magnitude between
    logged steps. The shear and contraction ratios divide by $H^2$, so the
    few CMB points where $H$ is near zero dominate the mean, and the 50
    points are redrawn every step. Taking the minimum of that series favours
    lucky draws over better geometry. Possible remedies, for selection only:
    evaluate $S$ on a fixed point set so changes reflect the model; use a
    median or trimmed mean over points to contain the near-zero-$H$ tail.
    Raising the point count (Task 4.11) helps once the tail is contained.

- **Task 4.14: Physical Constraints from Step Zero [FUTURE]**
  - **Scope:** the five inequality constraints, which are the weak energy
    condition, the three CMB priors (expansion, isotropy, homogeneity) and the
    present-day expansion band (Task 4.12). This task sets their violation
    measures, their Augmented Lagrangian starting state, the adaptive
    scheduler's bounds and the matching selection-score terms. The
    field-equation residual keeps its fixed weight `w_efe`.
  - **Current constraints and their problems:**
    - The CMB losses square absolute quantities in code units:
      $\max(0, -H_i)^2$, $\max(0, \sigma^2 - 10^{-5})^2$ and
      $\max(0, \sum(\partial_k g_{\mu\nu})^2 - 10^{-5})^2$. Their tolerances
      do not scale with the expansion rate the CMB measures them against,
      and squaring crushes a miss of $10^{-3}$ to $10^{-6}$.
    - Every constraint starts at $w = 1$, $\lambda = 0$ and gains strength only
      through the scheduler's $\times 5$ bumps at stalled checks, the first at
      step 1000. In the part-5 run the boundary contraction gave way only at
      step 7000, at $w = 15{,}625$.
    - The absolute density run (Task 4.12) left the boundary at
      $\sigma/H \approx 4$ with one axis contracting, under constraints that
      carried almost no force at that point in the run.
    - The selection score's shear and contraction ratios divide by $H^2$, so
      CMB points with $H$ near zero dominate it (Task 4.13, Score noise).
  - **Why the constraints stay Augmented Lagrangian:** blog post 3 showed that a
    fixed-weight energy-condition penalty settles on a compromise: the
    optimizer keeps a violation that helps the supernova fit and pays its fixed
    price. The multiplier $\lambda$ accumulates while any violation remains, so
    the price rises until the violation is zero; that run took `l_wec` to
    exactly 0. Blog post 4 showed the same equilibrium for a fixed penalty
    parameter, which the adaptive scheduler (section 5.2.4) addresses by
    escalating $w$ when progress stalls. Both hold for every constraint here:
    each is a physical condition, and a fixed weight would make it
    negotiable.
  - **Why they start with real weight:** the low starting state let the
    supernova fit establish itself before the physics engaged, because a
    physics-heavy start collapsed to Minkowski (section 5.2.1). Minkowski no
    longer costs nothing: the absolute density gives flat
    static space an EFE residual of $\kappa\rho_0$, `l_sn_cover` penalises
    unreachable supernovae, and the expansion band excludes $H \approx 0$. So
    each constraint starts at a weight that makes a full violation comparable
    to the data terms, and the Augmented Lagrangian keeps both of its jobs:
    $\lambda$ accumulates towards exact satisfaction, and $w$ escalates when
    progress stalls.
  - **Violation measures:** each is dimensionless and evaluated per point. The
    constraint is a linear hinge on its tolerance, $c = \max(0, f - \epsilon)$,
    averaged over the points.

    | constraint | points | $f$ | range | $\epsilon$ |
    | --- | --- | --- | --- | --- |
    | Shear | CMB boundary | $\sigma^2 / (H^2 + \sigma^2 + H_{floor}^2)$ | $[0, 1)$ | $10^{-10}$ |
    | Expansion | CMB boundary | $\sum_i \max(0, -H_i)^2 / (\sum_i H_i^2 + H_{floor}^2)$ | $[0, 1]$ | $10^{-10}$ |
    | Homogeneity | CMB boundary | $\sqrt{\sum_k \sum_{\mu\nu} (\partial_k g_{\mu\nu})^2}$ | $\ge 0$ | $10^{-5}$ |
    | WEC | EFE points | $\max(0, -R)$ | $\ge 0$ | $0$ |
    | Expansion band | observer, $t = 1$ | $\max(0, H_{lo} - H_{obs})/H_{lo} + \max(0, H_{obs} - H_{hi})/H_{hi}$ | $\ge 0$ | $0$ |

    - Shear and expansion: $f \approx (\sigma/H)^2$ and the squared
      contraction-to-expansion ratio while those are small, so
      $\epsilon = 10^{-10}$ is the CMB's $10^{-5}$ bound on the ratio. Both
      approach 1 when shear or contraction dominates the expansion.
      $H_{floor} = 10^{-4}$ code units ($\approx 0.03$ km s⁻¹ Mpc⁻¹) treats
      smaller rates as zero, which keeps the fractions and their gradients
      bounded on near-static geometry; isotropic static space scores 0.
    - Homogeneity: the metric perturbation amplitude per Gpc, against the
      Sachs-Wolfe $10^{-5}$. The metric components are of order 1, so an
      order-1 perturbation scores order 1.
    - WEC: the linear violation of blog post 3, which keeps a constant slope
      however large the violation. The first stage-one run used the bounded
      fraction $\max(0, -R) / (\max(0, -R) + \kappa\rho)$. With $|R|$ at
      100-300 times $\kappa\rho$ it sat near 1 wherever $R < 0$, so it
      counted negative-curvature points without measuring them, and its
      slope there, $\kappa\rho / (|R| + \kappa\rho)^2 \approx 5 \times 10^{-4}$,
      left the largest violations almost unpushed. Its penalty sits outside
      the spatial weighting of section 5.7 (see WEC outside the spatial
      weighting, below).
    - Expansion band: $H_{obs}$ is the comoving $H_{mean}$ at the observer,
      $(1, 0, 0, 0)$, from `get_cosmology_today()`; Task 4.12 gives the
      reason for a single point. $H_{lo}, H_{hi} = 60, 80$ km s⁻¹ Mpc⁻¹,
      converted to code units by an explicit factor ($0.2001$, $0.2669$). A
      static universe scores 1. The rest of the slice is unconstrained, so a
      local expanding region around the observer is an admissible solution;
      the supernovae across the light cone still have to fit it.
  - **Starting state:** $\lambda_0 = 0$ for every constraint. The penalty is
    $\lambda c + \tfrac{1}{2} w c^2$, and a satisfied constraint costs nothing
    at any weight.
    - CMB priors and expansion band: $w_0 = 100$. A full violation
      ($c = 1$) costs 50 at step 0, the same as `w_sn * l_sn` at
      $\chi^2/N = 5$.
    - WEC: $w_0 = 10$, the weight of blog post 3's run. Its violation is
      unbounded: mean $\max(0, -R)$ of 1-3, as in the first stage-one run,
      costs 5-45 at this weight.
  - **Scheduler bounds:** each constraint's floor is its own $w_0$, so decay
    returns a weight to its starting pressure and never below it, and a
    constraint that is met and then regresses meets that pressure at once.
    The floor is configuration, set per constraint in the trainer, and is not
    checkpointed. `MAX_W` $= 10^4$ for all. It is sized for violations of
    order 1: at the cap a violation of $0.1$ costs 50, the size of the data
    terms. The $10^6$ cap was sized for squared absolute violations, where a
    miss of $10^{-3}$ registered as $10^{-6}$. The check interval is
    unchanged.
  - **Selection score:** the shear and expansion terms use the same
    fractions in accuracy units: $f_\sigma / 10^{-10}$ and $f_e / 10^{-10}$.
    Homogeneity, EFE and WEC are unchanged. The
    expansion band adds $(\Delta H / \sigma_H)^2$, with $\Delta H$ the distance
    of $H_{obs}$ outside the band and $\sigma_H = 1.04$ km s⁻¹ Mpc⁻¹
    ($0.00347$ code units), the SH0ES uncertainty. The shear and expansion
    terms are bounded at $10^{10}$, which contains the near-zero-$H$ tail
    described under Score noise (Task 4.13); the per-step redraw of the CMB
    points remains.
  - **Implementation:**
    - `l_expand`, `l_shear`, `l_spatial` and `l_wec` keep their names with the
      definitions above. `l_h0` joins the loss terms and `h_today` ($H_{obs}$)
      the run metrics in `training_metrics.csv`. Penalty events stay in their
      own channel.
    - `ConstraintStates` gains `h0`. The change lands with a fresh run, so
      the `.meta` schema needs no migration.
    - Sections 4.3 and 5.2.4 describe the CMB forms and the scheduler.
  - **Tests:** expanding FLRW scores 0 on shear and expansion; contracting FLRW
    scores 1 on expansion; Kasner $p = (-\tfrac13, \tfrac23, \tfrac23)$ scores
    $\tfrac34$ on shear and $\tfrac19$ on expansion; Minkowski scores 0 on
    both; the band scores 0 inside, 1 at $H_{obs} = 0$ and its relative
    excess above; the scheduler decays to each constraint's own floor and
    caps at `MAX_W`; resume restores the `h0` state.
  - **Stage-one outcome (measured 2026-10-19, step ~3180):** training is
    stable from $w_0$ and the model stays off Minkowski without the
    expansion band. The boundary expands on every axis, close to
    isotropically ($\sigma/H \approx 0.07$ at the origin). `l_expand` is
    exactly 0 on about a third of logged steps after step 2500, `l_shear` is
    $\approx 0.01$, `l_spatial` fell from $0.47$ to $0.075$, `l_phys` from
    $0.28$ to $0.04$, and `l_sn_cover` touches 0. The best $S$ fell from
    $4.7 \times 10^9$ to $1.2 \times 10^8$. $\kappa\rho_0$ sits at the
    ceiling; the reported $\Omega_m$ fell from $\approx 600$ to
    $\approx 125$, so $H_{obs}$ rose to $\approx 0.011$, about 4% of $H_0$.
    `l_wec` has held at $\approx 1.05$ since step 600, through bumps to
    $w = 50$ (step 2000) and $w = 250$ (step 3000, $\lambda = 327$).
    By step ~16,500 (best $S = 3.6 \times 10^6$) the WEC weight has sat at
    the cap since step 7000 with $\lambda \approx 10^5$, and `l_wec` has
    fallen to $\approx 0.36$. The largest drop, $0.6 \to 0.35$, came between
    steps 13,000 and 15,000, when its own pressure barely changed and the
    homogeneity and shear weights rose. $H_{obs}$ stopped rising at $\approx 0.034$ (14% of
    $H_0$) around step 10,000.
  - **WEC outside the spatial weighting [COMPLETED]:**
    - **Mechanism:** the WEC penalty is added to the EFE residual before
      the per-point homoscedastic weighting
      $\tfrac12 e^{-2W} L + W$ (section 5.7). The optimum
      $W = \tfrac12 \ln L$ rises with $L$, the cost becomes
      $\approx \tfrac12 (1 + \ln L)$, and the gradient on the metric becomes
      $\tfrac12 \partial \ln L$, which no constant multiple of $L$ changes.
      The scheduler's bumps and the growth of $\lambda$ are absorbed by
      raising $W$ where WEC is violated. This is the lazy optimizer of blog
      post 3 ($\sigma \to \infty$), acting per point. The CMB terms sit
      outside the weighting and respond to their weights.
    - **Evidence:** at step 2450 the unweighted WEC penalty
      ($\lambda \bar c + \tfrac12 w \bar c^2$ with $\lambda = 65$, $w = 50$,
      $\bar c \approx 1.03$) is $\gtrsim 93$, while the total loss less the
      data and CMB terms leaves $\approx 3$ for EFE and WEC together, against
      $\tfrac12 (1 + \ln 93) \approx 2.8$. At step 3000 the bump to
      $w = 250$, $\lambda = 327$ would add $\approx 270$ unweighted; the loss
      rose from 13.4 to 17.8 and `l_wec` did not move.
    - **$W$ measured (best checkpoint, step 16,522):** $W$ is 3–6 over
      almost all of the sampled domain at every integer $t$ from $-4$ to 1,
      about 4.4 at the observer today, so the physics carries a factor
      $\tfrac12 e^{-2W} \approx 10^{-3}$ to $10^{-6}$. It is a smooth tilt
      across space that barely changes with $t$, with no feature at the
      observer's curvature lump or the negative-$R$ patches: the network
      suppresses regions, not points, which is why WEC moved at all. At
      balance $L = e^{2W} \approx 3000$, far above the EFE residual
      ($\approx 2 \times 10^{-3}$) and in the range of WEC's
      $\lambda c$. Sharing the weighting with WEC, the EFE residual's share
      of the gradient is divided by the same $L$, so the field equations are
      suppressed wherever WEC dominates $L$, which with a smooth $W$ is
      nearly everywhere.
    - **Change:** the per-point WEC penalty $\lambda c + \tfrac12 w c^2$ is
      averaged over the EFE points and added to the loss alongside the CMB
      penalties. The spatial weighting applies to the EFE residual alone.
    - **Diagnostic:** the spatial-weight row of the notebook's 2D maps
      (cell `c82d99f1`), at every integer $t$, on the best and latest
      checkpoints.
    - **Prediction:** with WEC out, each point's $L$ is the EFE residual
      alone, $\sim 10^{-3}$, and the balance point
      $W = \tfrac12 \ln L$ is $\approx -3$. $W$ should fall from 3–6 to
      below zero, weighting the field equations up. If $W$ stays high, the
      mechanism above is wrong and the weighting comes off the EFE residual
      as well.
    - **Resumption:** lands with a fresh run. A checkpoint's WEC state was
      accumulated under suppression, and applied unweighted it would
      dominate the loss.
    - **Outcome:** the prediction held, so the weighting stays on the EFE
      residual. $W$ was below zero from step 30 and fell to $-7$ to $-5.5$
      at the best checkpoint (step 9719). `l_wec` fell from 0.6 to
      $\approx 0.01$ at $w_0 = 10$ and to $2 \times 10^{-4}$ after its first
      bump. The logged EFE fell without plateau and the loss had no spikes.
  - **CMB shear and homogeneity at the cap (open):** in the same run the
    shear fraction held at $5 \times 10^{-3}$ to $10^{-2}$ from step 3500 at
    every weight, and the homogeneity term at $\approx 0.016$ from step
    7000. Both escalated to `MAX_W` (shear at step 10,000, homogeneity at
    11,000) without falling; with $\lambda$ rising by $\approx w c = 300$ per
    check, their penalties came to dominate the loss, `l_sn` rose from 0.49
    to 2–4 and `l_sn_cover` from $2 \times 10^{-5}$ to $\sim 10^{-2}$. The
    success criterion below fails on this point. Task 4.15 changes the
    homogeneity term's scale, so the diagnosis follows it.
  - **Sequencing:** stage one, then WEC outside the spatial weighting, then
    Tasks 4.15 and 4.16 together, then the expansion band.
    - Stage one: the violation measures, starting state, scheduler bounds
      and score terms for the four existing constraints, with a run to
      confirm that starting at $w_0$ trains stably. The case for a real
      starting weight rests partly on the expansion band, so this stage
      also shows whether the absolute density and coverage alone keep the
      model off Minkowski.
    - WEC outside the spatial weighting, on a fresh run, checked against
      the $W$ prediction.
    - Observer scale normalisation (Task 4.15) and the luminosity distance
      from the ray bundle (Task 4.16), on one fresh run. Until both land,
      the supernova term compares the data with a coordinate distance, so
      no run's metric can be read as a fit to the data.
    - The expansion band joins the set on the same mechanism. Its edges are
      fixed by local measurements. Landing it after Tasks 4.15 and 4.16
      shows what the supernovae alone do to $H_{obs}$.
  - **Success Criterion:** every constraint carries force from step 0;
    $H_{obs}$ enters $[0.200, 0.267]$; the CMB shear and expansion fractions
    fall towards tolerance without any weight reaching `MAX_W`; `l_sn` holds
    a genuine fit with `l_sn_cover` at zero.

- **Task 4.15: Observer Scale Normalisation, $a_0 = 1$ [COMPLETED]**
  - **Goal:** fix the overall spatial scale of the metric by the convention
    $\det h = 1$ at the observer today, $(1, 0, 0, 0)$. One coordinate unit
    is then 1 Gpc of proper length at the observer, and $\kappa\rho_0$ is the
    observer's density today.
  - **Problem:** nothing in the parameterisation or the loss fixes the
    overall size of $g_{ij}$. Multiplying every proper spatial length by $c$,
    $g \to S g S$ with $S = \mathrm{diag}(1, c, c, c)$, leaves $H$, the shear
    fraction and the contraction fraction unchanged, multiplies $R$ by
    $c^{-2}$ and the local density $\kappa\rho_0 / \sqrt{\det h}$ by
    $c^{-3}$. The value of $a_0$ is a choice of units, and the code already
    assumes $a_0 = 1$ in two places:
    - Density (section 5.4): $\kappa\rho_0$ is the density where
      $\det h = 1$. Its bounds bound the observer's density only when
      $\det h_{obs} = 1$, and the reported $\Omega_m$ uses $\kappa\rho_0$.
    - Distances: `get_luminosity_distance` returns $(1 + z)\,r$ with $r$ the
      coordinate radius in Gpc, which is the flat FLRW $(1 + z)\,a_0 r$ at
      $a_0 = 1$. Task 4.16 computes the distance from the ray bundle as a
      proper length, and $a_0 = 1$ sets its unit: one code unit is 1 Gpc of
      proper length at the observer.

    Two loss terms reward a smaller overall scale:
    - EFE: the residual is multiplied by $\gamma = \det g_{ij}$ (section 5.6),
      so the trained mean square scales as $c^{8}$ while the unweighted
      residual scales as $c^{-4}$.
    - Homogeneity: $\sqrt{\sum (\partial_k g_{\mu\nu})^2}$ is not normalised
      by the metric and scales roughly as $c$.
  - **Measured:** the latest checkpoint of the WEC-outside-the-weighting run
    (step 9390), rescaled by $c$ and evaluated on CPU at that checkpoint's
    multipliers and weights (512 EFE points, 512 CMB points, 96 supernovae):

    | $c$ | logged EFE | unweighted EFE | EFE + WEC term | CMB penalties | SN | total |
    | --- | --- | --- | --- | --- | --- | --- |
    | 0.8 | $4.5 \times 10^{-7}$ | 75.6 | $-6.01$ | 2.31 | 0.81 | $-2.89$ |
    | 1.0 | $2.6 \times 10^{-6}$ | 30.0 | $-5.84$ | 2.58 | 0.71 | $-2.55$ |
    | 1.25 | $1.6 \times 10^{-5}$ | 12.5 | $-4.63$ | 3.00 | 1.93 | $-0.30$ |

    $H_{obs}$ is $0.0305$ at every $c$, and $R_{obs}$ scales exactly as
    $c^{-2}$. The supernova term is lowest near the current scale, and the
    total is still lower at $c = 0.8$. In training, $a$ at the observer has
    fallen in every run with scale-factor plots: 0.73 at the part-5 best,
    0.41 at the part-6 best, and 0.47, 0.30, 0.27 at steps 5129, 7770 and
    9390 of the WEC run, while `l_sn` held near 0.5. At step 9390 the
    observer's density is $\kappa\rho_0 / a^3 = 2.43$, 52 times the
    ceiling; at the part-6 best it was about 14 times. $R\,a^2$ at the
    observer stayed at 15–17 across these checkpoints: the curvature lump
    there has the same proper geometry at each scale.
  - **Change:** the normalisation is part of the metric. `MetricNN.__call__`
    evaluates the raw network at the observer, takes
    $s = (\det h_{obs})^{-1/6}$ and returns $S g S$ with
    $S = \mathrm{diag}(1, s, s, s)$. This map takes $h \to s^2 h$, so
    $\det h_{obs} = 1$ exactly, and a uniform rescaling of the raw output
    leaves the returned metric unchanged. The factor is constant in the
    coordinates, so the metric's coordinate derivatives carry it as a
    constant and the Ricci evaluation adds one forward pass at the observer
    and no derivative work. Section 4.1 builds the Lorentzian signature into
    the parameterisation for the same reason: a convention enforced by a
    penalty would be negotiable against the terms above.
  - **Open:** with $a_0$ fixed, $\gamma$ is $(a / a_0)^6$, the relative
    weighting of section 5.6, and the homogeneity term depends on the scale
    through $a(t{=}{-4}) / a_0$. Both still favour a smaller $a / a_0$ away
    from the observer, which is more expansion since that point and is
    constrained by the supernova redshifts. Whether the logged EFE and the
    unweighted residual fall together shows whether that pull is material,
    and Task 4.17 acts on the result.
  - **Tests:** $\det h = 1$ at $(1, 0, 0, 0)$ for a randomly initialised
    model; scaling the raw network's spatial block by any $c$ leaves the
    returned metric unchanged; an FLRW metric with $a(1) \neq 1$ is returned
    with $a(1) = 1$ and the same $H(t)$; the signature stays $(-, +, +, +)$.
  - **Resumption:** lands with Task 4.16 on a fresh run. Every existing
    checkpoint has $a_0 \neq 1$, and normalising it changes its metric.
  - **Success Criterion:** `a_mean` at $t = 1$ in the notebook reads 1;
    the observer's density lies in $[\kappa\rho_b, \kappa\rho_m]$.

- **Task 4.16: Luminosity Distance from the Ray Bundle [COMPLETED]**
  - **Goal:** each supernova's $d_L$ is the model's own luminosity
    distance, valid in any metric the network produces.
  - **Problem:** `src/physics/geodesics.py` uses FLRW forms in four places.
    They have been there since the ray tracer was written:
    - Distance: `get_luminosity_distance` returns $(1 + z)$ times the
      coordinate radius of the ray's endpoint, which is the flat FLRW $d_L$
      at $a_0 = 1$. In any spacetime $d_L = (1 + z)^2 d_A$ (Etherington
      reciprocity), and the area distance $d_A$ is set by the focusing and
      shear of the ray bundle (the Sachs optical equations). Matter in the
      beam focuses it through $R_{\mu\nu} k^\mu k^\nu$, and nearby structure
      shears it through the Weyl tensor. The coordinate radius carries
      neither, so inhomogeneity reaches the distances only through the
      redshift and the ray's path.
    - Redshift: `get_redshift` uses
      $u_\mu = (-\sqrt{-g_{00}}, 0, 0, 0)$, which is the comoving observer
      only when $g_{0i} = 0$. Comoving dust has
      $u_\mu = g_{\mu 0} / \sqrt{-g_{00}}$, the 4-velocity the stress-energy
      tensor and the kinematics already use.
    - Sky direction: `make_null_direction` uses the RA/Dec unit vector as a
      coordinate direction at the observer. That vector is a direction on
      the sky only when $h_{obs}$ is isotropic.
    - BAO: `get_bao_distances` takes $D_M$ as the coordinate radius and
      $D_H$ as $1/H$ at the spatial origin at the source's time. `l_bao`
      logs 0 in current runs.

    Task 3.3 verifies the ray tracer against FLRW, where the coordinate
    form is exact, so it passes by construction.
  - **Measured:** the WEC run's best checkpoint (step 9719) has
    $a_0 = 0.262$ and $R \approx 240$ at the observer. Near the observer its
    proper distances are $a_0$ times the coordinate ones, about 3 magnitudes
    in $\mu$, before any focusing by that curvature. Every run's supernova
    fit, including that run's $\chi^2/N \approx 0.5$, is a fit of the
    coordinate form.
  - **Change:**
    - Observer frame: an orthonormal tetrad at $(1, 0, 0, 0)$ with
      $e_0 = u$, the comoving 4-velocity, and the spatial triad $L^{-T}$
      from the Cholesky factor $h_{obs} = L L^T$, which lies along the
      coordinate axes when $h_{obs}$ is isotropic. RA/Dec give the unit
      vector $m$ towards the source, orthogonal to $u$, and the initial null
      vector $k = u - m$ has unit frequency, $-u \cdot k = 1$, for the
      observer (`make_null_direction`).
    - Jacobi map: two deviation vectors $\xi_A$ ride alongside the ray, one
      for each of two orthogonal angles $\theta^A$ on the observer's sky
      about the source. Each starts at zero with rate
      $\partial k / \partial \theta^A$ and evolves by the geodesic flow
      linearised about the ray (`jax.linearize`), which is the geodesic
      deviation equation in coordinates. The curvature enters through the
      derivatives of the Christoffel symbols, so no Riemann tensor is
      formed. The state grows from 8 to 24 components.
    - Area distance: on the screen at each point, the 2-plane orthogonal to
      the comoving $u$ and to the ray's direction $d$ in $u$'s rest frame,
      the deviations have inner products
      $G_{AB} = \xi_A \cdot \xi_B + (u \cdot \xi_A)(u \cdot \xi_B) - (d \cdot \xi_A)(d \cdot \xi_B)$,
      and $d_A^2 = \sqrt{\det G}$ (`area_distance`). Then
      $d_L = (1 + z)^2 d_A$ in code units of Gpc. A lumpy metric's $d_L$
      need not be monotonic, so states past the end of a ray carry its last
      finite distance.
    - Redshift: $1 + z = (u \cdot k)_{src} / (u \cdot k)_{obs}$ with the
      comoving $u_\mu$, shift included.
    - BAO: $D_M = (1 + z)\,d_A$ along each ray. $D_H$ keeps its form.
  - **Interactions:**
    - With Task 4.15: $d_A$ is a proper length, so a uniform rescaling of
      $g_{ij}$ changes every $d_L$ and the supernovae measure the scale. The
      convention $a_0 = 1$ fixes the unit, and the density bound needs it in
      any case.
    - With the expansion band: as $z \to 0$, $d_A \to z / H$ at the observer
      (Kristian–Sachs, sky-averaged), so the low-redshift supernovae
      constrain the observer's expansion directly. The band's edges are
      local measurements and stay fixed.
  - **Diagnostic:** the supernova $\chi^2$ of the WEC run's best checkpoint
    under the new distance measures how far that fit was from the model's
    own luminosity distances.
  - **Tests:**
    - Flat FLRW with $a(1) = 1$: $d_L$ matches the analytic value for
      $z \in [0.01, 2.3]$ to the ODE tolerance.
    - A constant anisotropic $h$ at the observer, from a linear change of
      spatial coordinates, leaves $d_L$ and $z$ unchanged in every sky
      direction.
    - FLRW resliced by $t = t' - c\,x$, which keeps every comoving worldline
      and gives the metric a shift, gives the same $z$ and $d_L$ as without
      it.
  - **Resumption:** lands with Task 4.15 on a fresh run.
  - **Success Criterion:** the three tests pass, and `l_sn` is computed from
    the area distance in training.

- **Task 4.17: Scale-Independent Residual Weighting [FUTURE, conditional]**
  - **Goal:** the EFE and homogeneity terms give no reward for a smaller
    $a / a_0$ away from the observer.
  - **Trigger:** the measurement in Task 4.15's open item. On checkpoints
    of the first run with Tasks 4.15 and 4.16, evaluate the EFE residual
    over the collocation points with and without $\gamma$. If both fall
    together, the pull is immaterial and this task closes unused. If the
    logged EFE falls while the unweighted residual holds or rises, the
    model is lowering the trained number through $a / a_0$, and the task
    goes ahead.
  - **Change:** decided by which term carries the pull: the $\gamma$
    weighting of section 5.6, the homogeneity term's dependence on the
    metric's scale, or both.
  - **Success Criterion:** the logged EFE and the unweighted residual fall
    together.

---

## 5. Advanced Training Dynamics: Batching & Gradient Balancing

To transition the PINN training loop from a prototype to a production-grade
optimization engine, we must address the computational complexity of the data
loss and the inherent competition between the physics and observational
objectives.

### 5.1 Supernova Batching Design

Rather than evaluating all $1,701$ supernovae in the Pantheon+ dataset at each
training step—which requires solving $1,701$ separate null geodesic ODEs
backwards through time—we introduce stochastic batching of the observational
data.

- **JAX-Compatible Sampling:** We define a static batch size $B$ (e.g.,
  $B = 128$). At each step, we use `jax.random.choice` to sample $B$ indices
  from the dataset. By keeping the batch size constant, JAX's compilation engine
  (XLA) avoids recompilation.
- **The Stochastic Gradient Advantage:** Evaluating a subset of geodesics
  introduces stochastic noise into the observational loss gradient
  $\nabla_\theta \mathcal{L}_{Data}$. While this noise requires a slightly
  smaller base learning rate, it acts as a regularizer. The resulting "gradient
  thermal bath" helps the network parameters jump out of the shallow local
  minima and coordinate traps that characterize the unconstrained 10-dimensional
  metric landscape.
- **Implementation Details:**
  - `batch_z, batch_mu, batch_err = select_batch(sn_data, indices)`
  - The loss is normalized by the batch size $B$ to maintain consistent gradient
    scaling across different choices of $B$.

### 5.2 Hyperparameter Tuning & Gradient Balancing

Physics-Informed Neural Networks often fail when the gradients of the physics
loss ($\mathcal{L}_{Physics}$) and the data loss ($\mathcal{L}_{Data}$) have
vastly different magnitudes. If one dominates, the optimizer will satisfy it at
the absolute expense of the other.

#### 5.2.1 Biasing Towards Physics (The Cosmological Constraint)

Because a metric that fits the supernova data but violates Einstein's Field
Equations is physically meaningless, we must bias the optimization toward
satisfying General Relativity. However, if the physics loss weight $w_{Physics}$
is set too high, the model quickly collapses into the trivial vacuum Minkowski
metric (which has exactly zero curvature and zero physics loss) and refuses to
ever move toward fitting the expansion of the universe. To prevent this, we
propose two methods:

#### 5.2.2 Option 1: Dynamic GradNorm (Gradient Norm Balancing)

We dynamically adjust the loss weights $w_i(t)$ at each training step $t$ to
ensure the gradients of the physics and data losses remain in a physical
proportion.

- **The Algorithm:**
  1.  At step $t$, compute the gradients of the model's final layer parameters
      with respect to each individual loss component:
      $G_{phys}(t) = \|\nabla_{\theta_{last}} (w_{phys} \mathcal{L}_{phys})\|_2$
      and
      $G_{data}(t) = \|\nabla_{\theta_{last}} (w_{data} \mathcal{L}_{data})\|_2$.
  2.  Define a target ratio $\alpha$ (e.g., $\alpha = 10.0$ to bias the
      optimizer towards physics).
  3.  Compute the desired weight updates to bring the gradient ratio close to
      $\alpha$:
      $$w_{phys}(t+1) = (1 - \beta) w_{phys}(t) + \beta \left( \alpha \frac{\bar{G}(t)}{G_{phys}(t)} \right)$$
      where $\bar{G}(t)$ is the mean gradient norm, and $\beta$ is a momentum
      hyperparameter (e.g., $0.1$).
- **Tradeoffs:** Automatically prevents either loss from "drowning out" the
  other, stabilizing training across the entire 10,000-step cycle. However,
  computing separate gradients for each loss component adds slight
  backpropagation overhead.

#### 5.2.3 Option 2: Augmented Lagrangian (Hard Boundary Constraints)

We reformulate the training as a constrained optimization problem, treating the
Einstein Field Equations as a hard constraint:
$$\min_{\theta} \mathcal{L}_{Data}(\theta) \quad \text{subject to} \quad \mathcal{L}_{Physics}(\theta) \le \epsilon$$
We optimize the Augmented Lagrangian:
$$\mathcal{L}_{Aug}(\theta, \lambda) = \mathcal{L}_{Data}(\theta) + \lambda \mathcal{L}_{Physics}(\theta) + \frac{\mu}{2} \mathcal{L}_{Physics}(\theta)^2$$
Where:

- $\lambda$ is the Lagrange multiplier, updated after every epoch/batch step:
  $\lambda \leftarrow \lambda + \mu \mathcal{L}_{Physics}(\theta)$.
- $\mu$ is the penalty parameter, which increases over training to penalize
  constraint violations more severely.
- **Tradeoffs:** Guarantees that the final converged metric strictly satisfies
  General Relativity. However, it requires careful tuning of $\mu$'s growth rate
  to prevent numerical instability.

#### 5.2.4 Adaptive Penalty Scheduling

To automate the tuning of the Augmented Lagrangian penalty parameter $\mu$
(e.g., `w_wec`), we implement an Adaptive Penalty Schedule in the outer Python
training loop. This prevents the constraint from stalling in the late stages of
training when the violation becomes extremely small.

- **The Algorithm:**
  1. Every constraint starts at its own $\mu_0$ with $\lambda = 0$ (Task
     4.14). The CMB priors start at $\mu_0 = 100$: their violations are
     fractions of order 1 at most, so a full violation costs
     $\tfrac12 \mu_0 = 50$ from step 0, on a par with the supernova term.
     WEC, whose violation $\max(0, -R)$ is unbounded, starts at
     $\mu_0 = 10$.
  2. Define a check interval $N_{check}$ (`adaptive_check_interval`, 1000
     steps).
  3. At step $t = k \cdot N_{check}$, compare the current exponential moving
     average of the violation $\mathcal{L}_{k}$ against the violation at the
     previous check $\mathcal{L}_{k-1}$.
  4. If the violation has not decreased by a sufficient margin
     ($\mathcal{L}_{k} > \tau \mathcal{L}_{k-1}$, $\tau = 0.95$), multiply the
     penalty parameter by $\gamma = 5$. If it has fallen below half
     ($\mathcal{L}_{k} < 0.5 \mathcal{L}_{k-1}$), divide it by 2.
  5. Bound $\mu$ to $[\mu_0, 10^4]$: decay returns it to its starting
     pressure and never below. The cap is sized for violations of order 1,
     where a violation of $0.1$ at the cap costs 50.
- **Implementation:** Because $\mu$ (e.g., `w_wec_val`) is passed to the
  JIT-compiled `step` function as a dynamic `jnp.ndarray`, we can seamlessly
  multiply it by $\gamma$ in the outer Python loop without triggering
  recompilation. This system will natively scale to handle multiple constraints
  (WEC, CMB Expansion, CMB Isotropy) independently.

### 5.3 Parameter Sensitivity & Gradient Boosting (Optax Multi-Transform)

A major bottleneck when optimizing deep network parameters alongside single
physical scalars (such as the trainable matter density parameter $\Omega_m$) is
gradient scale imbalance.

- **The Problem:** The network contains thousands of weights that determine the
  complex curvature profiles, making the loss highly sensitive to their changes.
  When `optax.clip_by_global_norm(1.0)` is applied, it divides the entire
  gradient vector by the global norm. Consequently, the gradient of the single
  scalar $\Omega_m$ is scaled down to near-zero, causing it to move extremely
  slowly over training steps.
- **The Solution (Parameter Groups):** Rather than raising the global learning
  rate (which destabilizes the network), we use `optax.multi_transform` to split
  the parameters into separate optimization groups.
- **Implementation:** The metric network parameters are assigned to the `"mlp"`
  group (with global norm clipping and standard scheduling), while the
  $\Omega_m$ parameter is assigned to the `"omega"` group. The `"omega"` group
  receives a dedicated `optax.adam` optimizer with a raw `100.0x` multiplier
  applied to the base learning rate schedule, completely bypassing the global
  norm clipping bottleneck.
- **Physical Justification:** The matter density represents a global physical
  constant that must adjust rapidly early in training to broadly balance the
  EFE, while the spatial metric network weights must adjust slowly and smoothly
  to avoid introducing singular coordinate ripples.

### 5.4 Trainable Matter Density Parameterization & Dynamic Floor

The model's only trainable physical scalar is the absolute matter density
$\kappa\rho_0$. It is independent of the expansion rate, so a geometry cannot
remove its matter by not expanding.

- **Bounds, as absolute densities:** $\kappa\rho = 8\pi G\rho/c^2$ in code
  units of Gpc⁻². No expansion rate enters either bound.
  - Floor: the baryons, $\rho_b = 4.17\times10^{-28}$ kg/m³, so
    $\kappa\rho_b \approx 0.0074$. Big Bang nucleosynthesis fixes the
    baryon-to-photon ratio from deuterium abundances, and the measured CMB
    temperature fixes the photon number density.
  - Ceiling: total matter, $\rho_m = 2.63\times10^{-27}$ kg/m³, so
    $\kappa\rho_m \approx 0.047$. This end is softer: its tightest value comes
    from fits within $\Lambda$CDM, with cluster and dynamical estimates in a
    similar range.

  Once $H(t{=}1)$ is in the observed band these correspond to
  $\Omega_m \approx 0.04$–$0.26$. Bounding $\Omega_m$ directly would re-couple
  the density to $H$ and bring back the $H \to 0 \Rightarrow \rho \to 0$
  fixed point.
- **Parameterization:** `kappa_rho_0_raw` lives in log space above the floor,
  $\kappa\rho_0 = \kappa\rho_b\,e^{\text{raw}}$, so its optimizer group's step
  of ~$10^{-2}$ moves the density by ~1%. It starts at the floor (baryons
  only) and rises if the field equations need more matter.
- **PGD clipping:** the optimizer updates `kappa_rho_0_raw` freely, then
  `clip_matter_density` clips it to $[0, \ln(\kappa\rho_m/\kappa\rho_b)]$
  after each step. Clipping avoids the landscape warping that
  shifted-softplus bounds introduced.
- **Dilution:** comoving dust conserves $\rho\sqrt{\det h}$, with
  $h_{ij} = g_{ij} - g_{0i}g_{0j}/g_{00}$ the spatial metric in its own rest
  frame, so the local density is $\kappa\rho_0/\sqrt{\det h}$. The metric
  is normalised to $\det h = 1$ at the observer today (Task 4.15), so
  $\kappa\rho_0$ is the observer's density and its bounds apply there.
- **Reported:** $\Omega_m = \kappa\rho_0 / 3H_{mean}(t{=}1)^2$, derived at the
  model's own expansion rate. It runs far outside $[0, 1]$ while $H$ is
  unphysical.

### 5.5 EFE Coordinate Sampling & Active Redshift Region Prioritization

To prevent the metric network from exploiting unobserved gaps in the coordinate
space to build unphysical coordinate singularities (e.g. "gravitational redshift
pumps" designed to fake expansion history), we implement a hierarchical sampling
scheme for the training domain.

- **The Problem (Adversarial Singularities):** Under sparse uniform sampling of
  coordinate time $t$, a narrow coordinate singularity (width $\approx 0.1$) can
  easily hide in the gaps between points. The network exploits this to generate
  the required observational redshift in a single violent contraction-expansion
  bounce. Furthermore, the early universe ($t \in [-4.0, -2.5]$) exhibits
  extremely steep, sharp geometric warping that will be mathematically aliased
  if not resolved with high point density.
- **The Solution (Massive Resolution Bumping & Density Ratio):** We utilize a
  dense, two-tiered sampling strategy to fully utilize GPU compute headroom
  while mathematically oversampling the Supernova domain:
  - **Active Supernova Region:** We pack **800 points** into
    $t \in [-2.5, 1.0]$. With a span of 3.5, this yields $\approx 228$ points
    per unit redshift. This heavily anchors the network to the Pantheon+
    dataset.
  - **Deep Past Extension:** We pack **200 points** into $t \in [-4.0, -2.5]$.
    With a span of 1.5, this yields $\approx 133$ points per unit redshift. This
    is still a massive absolute resolution bump (preventing deep-past aliasing)
    but preserves the mathematical priority of the active region.

### 5.6 Non-Linear Coordinate Mappings & Mixed Tensor Formulation

Transitioning from a linear time mapping ($t=-z$) to a non-linear mapping (e.g.,
$t=5a-4$) introduces severe optimization challenges for the PINN:

- **Coordinate Variance of the EFE:** The covariant Einstein tensor $G_{\mu\nu}$
  scales heavily with the coordinate choice. Under $t=5a-4$, the covariant
  residual $G_{tt} - T_{tt}$ is artificially amplified at high redshifts
  compared to $t=-z$.
- **The Mixed Tensor Solution:** To decouple the optimization landscape from the
  arbitrary choice of $t$, we compute the EFE residual using **Mixed Tensors**:
  $G^\mu_\nu - 8\pi G T^\mu_\nu = 0$. For a dust universe, $T^t_t = -\rho$
  exactly, rendering the target perfectly coordinate-invariant.
- **The Physical Sampling Trap:** Even with mixed tensors, the physical density
  $\rho$ scales as $a^{-3}$. At $t=-3.5$ ($z=9$), the squared EFE residual is
  $10^6$ times larger than today. Because the loss is calculated as a
  `jnp.mean`, increasing the _density_ of sampled points in the early universe
  will cause this $10^6$ magnitude to completely overwhelm the optimizer,
  destroying late-time metrics (like the WEC penalty).
- **The Conclusion:** It is physically impossible to safely extend the sampling
  domain to high redshifts (e.g., to fully cover the $z \approx 2.3$ Supernova
  data) without implementing **Dynamic Gradient Balancing** (Task 4.7) or
  Relative Weighting (e.g., dividing the residual by $a(t)^4$). Without this
  balancing, the massive physical gradients from the early universe will
  inevitably drown out the late-time WEC and observational losses.

### 5.7 Spatially Adaptive Weighting (Solving the Coordinate Imbalance)

#### The Problem: Non-Linear Mapping Imbalance

While global parameters and physical $\gamma$ volume scaling correctly normalize
the _physical_ density ($a^{-3}$) across the universe, we still face an internal
gradient imbalance due to the non-linear time coordinate mapping $t(z)$. Regions
of high mathematical stiffness (e.g., early universe boundaries or sharp
inflection points) produce massive gradients in the WEC and EFE residuals.
Because the loss is calculated as a global average (`jnp.mean`), these stiff
local regions completely dominate the loss landscape, preventing the network
from fine-tuning the easier regions (like the late universe, where the Supernova
data sits) and preventing the WEC penalty from fully converging to `0.00`.

#### The Solution: Learned Spatial Weights $W(t)$

Instead of global scalar weights, we introduce a continuous, Learned Spatial
Weighting network. By evaluating a tiny auxiliary neural network
$W(t) \to \mathbb{R}$ at every collocation point, we allow the optimizer to
dynamically learn a spatial attention curve.

Using the homoscedastic uncertainty formulation _per-point_:

$$\mathcal{L}_{phys} = \frac{1}{N} \sum_{i=1}^N \left( \frac{1}{2} e^{-2 W(t_i)} \cdot \mathcal{L}_{residual}(t_i) + W(t_i) \right)$$

1. **Dynamic Suppression:** If the EFE residual is massive at a specific
   boundary $t=-3.5$, the optimizer will locally increase $W(-3.5)$ to suppress
   that stiff gradient.
2. **Spatial Curriculum Learning:** By suppressing the hardest coordinate
   domains, the network is free to solve the PDE in the "easy" regions first. As
   the easy regions converge, their gradients drop, and the network can slowly
   decrease $W(t)$ in the hard regions to tackle them sequentially, acting as a
   completely automated spatial curriculum.

$\mathcal{L}_{residual}$ is the EFE residual alone, scaled by `w_efe`. The WEC
Augmented Lagrangian penalty is averaged over the same points and added outside
the weighting (`physics_loss` in `src/training/loss.py`): the optimal
$W = \tfrac12 \ln \mathcal{L}$ makes the weighted gradient
$\tfrac12 \partial \ln \mathcal{L}$, which no constant multiple of
$\mathcal{L}$ changes, so inside the weighting the penalty's weight and
multiplier would have no effect (Task 4.14).

#### Architectural Implementation Details

- **The Network:** A very lightweight Multi-Layer Perceptron (e.g., 2 hidden
  layers of 16 neurons) appended as a submodule within `MetricNN`.
- **Input:** Because the network has discovered an _inhomogeneous_ cosmology
  (e.g., a spatial curvature dipole), we have categorically broken the Bianchi
  Type I homogeneity assumption. Therefore, the spatial weight network must take
  the full 4D coordinate vector $(t, x, y, z)$ as input to map out a complete 4D
  spatiotemporal attention mask.
- **Optimization:** The $W(t, x, y, z)$ network parameters are updated
  simultaneously with the main metric weights. Because it is evaluated inside
  the spatial `jax.vmap` batch, it adds virtually zero computational overhead.

### 5.8 Dynamic Learning Rate Scheduling (Coupled Decay)

#### The Problem: Gradient Shocks in Rigid PDEs

Standard learning rate schedules (like cosine decay) are purely step-based.
However, Physics-Informed Neural Networks solving rigid PDEs (like the Einstein
Field Equations) often encounter sudden, steep cliffs in the loss landscape.
When the Augmented Lagrangian constraints trigger a penalty bump (e.g.,
$w_{penalty} \leftarrow \gamma w_{penalty}$), the physics loss gradients
instantly multiply. If the global learning rate remains constant during this
penalty bump, the optimizer violently overshoots the thin valid manifold,
causing explosive `NaN` errors or the "Lazy Optimizer" trap.

#### Best Practice for PINNs: Penalty-Coupled Decay

While standard deep learning uses loss-plateau detection (`ReduceLROnPlateau`),
calculating and monitoring individual gradient norms for every loss component at
every step in JAX is computationally prohibitive. The most robust,
computationally "free" approach for Augmented Lagrangian PINNs is
**Penalty-Coupled Decay**.

Instead of purely monitoring the loss or global gradient norm (which Optax
already loosely manages via `clip_by_global_norm`), we tie the learning rate
directly to the constraint scheduler:

1.  **Reactive Drop:** Whenever the `ConstraintManager` triggers a penalty bump
    for _any_ physical constraint (indicating the physics residual is stubbornly
    high), we immediately multiply the global learning rate by a decay factor
    (e.g., $0.5$). This allows the network to carefully "tiptoe" through the
    suddenly stiffer geometry without tearing the spacetime metric.
2.  **Plateau Decay:** As a secondary fallback, if the total loss plateaus for
    an extended period (e.g., 1000 steps) _without_ triggering a penalty bump,
    we decay the learning rate to naturally anneal the optimizer into the local
    minima.
3.  **Implementation:** This requires refactoring the compiled `step()` function
    in JAX to accept a dynamic `lr_multiplier` scalar as an explicit argument,
    decoupling the optimizer from a rigid step-based Optax schedule.

---

## 6. Testing & Validation

- **Unit Tests:** Verify $g_{\mu\nu}$ symmetry and signature.
- **Symmetry Tests:** Check if the learned metric respects the requested Bianchi
  Type I spatial symmetries.
- **Performance:** Benchmark training speed. The RTX 5070 must handle
  $\sim 10^5$ collocation points per minute.
