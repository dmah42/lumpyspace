# Comprehensive Technical Design: Inhomogeneous Cosmology PINN

## 1. Executive Summary
This document defines the architecture, physical foundation, and implementation strategy for a Physics-Informed Neural Network (PINN) designed to reconstruct the spacetime metric $g_{\mu\nu}$ of the universe.

**The Goal:** To model a universe that relaxes the strict homogeneity and isotropy assumptions of the FLRW metric, using **Bianchi Class I** models as a baseline. We aim to determine if anisotropic expansion and local backreaction can resolve the Hubble Tension and account for Dark Energy effects without a cosmological constant ($\Lambda$).

**The Approach:** A JAX-based PINN acts as a universal function approximator for $g_{\mu\nu}(t, x, y, z)$, constrained by the Einstein Field Equations (EFE) and grounded in Supernovae distance data.

---

## 2. Theoretical Foundation (The "Why")

### 2.1 The Breakdown of FLRW
The Friedmann-Lemaître-Robertson-Walker (FLRW) metric assumes a "Mean Field" universe. By averaging out structure before solving Einstein's equations, we ignore the non-linear "backreaction" terms ($\mathcal{Q}_D$). In General Relativity, **averaging and evolution do not commute**: $G_{\mu\nu}(\langle g \rangle) \neq \langle G_{\mu\nu}(g) \rangle$.

### 2.2 The Target: Bianchi Type I & Shear ($\sigma$)
We transition to **Bianchi Type I**, the simplest anisotropic generalization of FLRW. It allows for a "Hubble Tensor" where expansion rates $H_x, H_y, H_z$ can differ. This introduces the **shear tensor** $\sigma_{\mu\nu}$, which acts as an effective energy density in the Raychaudhuri equation:
$$\dot{\theta} + \frac{1}{3}\theta^2 + \sigma^2 - \omega^2 = -4\pi G (\rho + 3p)$$
Our model will test if a non-zero $\sigma^2$ at late times can mimic the acceleration attributed to $\Lambda$.

---
## 3. Implementation Stack & Structure

### 3.1 Software Stack
...

### 3.2 Engineering Standards & Style Guide
To ensure maintainability and consistency, the following standards are enforced:
*   **Line Length:** Strict 80-character maximum.
*   **Indentation:** 2-space indentation (no tabs).
*   **Code Quality:** Enforced via `ruff` and `pre-commit`.
*   **Testing:** 100% coverage required for core tensor math.
*   **Type Hinting:** Mandatory for all functions (JAX `Array` types specified).

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
The network $f_\theta: (t, x, y, z) \to \mathbb{R}^{10}$ outputs the unique components of the symmetric metric $g_{\mu\nu}$.
*   **Signature Enforcement:** We ensure a Lorentzian signature $(-, +, +, +)$ by:
    *   $g_{00} = - \exp(\text{NN}_0)$
    *   $g_{ii} = \exp(\text{NN}_i)$
*   **Activation Function:** `SiREN` (Sine activations). Since EFE requires $C^2$ continuity, Sine functions ensure that high-order gradients do not vanish or become discontinuous.

### 4.2 The Multi-Probe Loss Function
$$\mathcal{L}_{Total} = w_P \mathcal{L}_{Physics} + w_D \mathcal{L}_{Data} + w_{Reg} \mathcal{L}_{Reg}$$

1.  **Physics Loss ($\mathcal{L}_{Physics}$):**
    Computes the residual of $G_{\mu\nu} - 8\pi G T_{\mu\nu} = 0$.
    *   We use `jax.jacfwd(jax.jacrev(metric))` to get the second derivatives for the Ricci scalar.
    *   $T_{\mu\nu}$ is modeled as a perfect fluid $T_{\mu\nu} = (\rho+p)u_\mu u_\nu + p g_{\mu\nu}$.
2.  **Data Loss ($\mathcal{L}_{Data}$):**
    Requires integrating the geodesic equation for light: $\frac{d^2 x^\mu}{d\lambda^2} + \Gamma^\mu_{\alpha\beta} \frac{dx^\alpha}{d\lambda} \frac{dx^\beta}{d\lambda} = 0$.
    *   We solve this using `Diffrax` to find the luminosity distance $d_L(z)$ and compare against the **Pantheon+** dataset.
3.  **Regularization ($\mathcal{L}_{Reg}$):**
    Enforces the **Cosmological Principle** as a high-redshift prior ($z > 10$), penalizing $\sigma > 0$ in the early universe to remain consistent with CMB observations.

### 4.3 The CMB Isotropy Paradox (Open Research Question)
**The Tension:** This project aims to "relax strict homogeneity and isotropy assumptions." However, Task 4.2 originally proposed a "high-redshift prior" that forces the model toward isotropy at $z > 10$. This risks "baking in" the FLRW result we are trying to test.

**The Investigation:**
- **The Question:** Does the metric *become* isotropic at high redshift, or does it only *appear* isotropic in our observations?
- ** Nuanced Constraint:** Instead of a hard penalty for any $\sigma > 0$ (shear), we should investigate implementing a "CMB-consistent bound." This would allow the model to explore anisotropic solutions as long as their observational signature at the surface of last scattering remains within the $10^{-5}$ fluctuations observed by Planck/WMAP.
- **Future Task:** We must determine how to map 4D metric fluctuations to the CMB temperature power spectrum to create a non-biased high-redshift constraint.

---

## 5. Phased Implementation Roadmap
## 5. Exhaustive Phased Implementation Roadmap

### Phase 1: The FLRW Control Baseline (Verification & Calibration)
**Goal:** Prove the 4D PINN architecture can perfectly recover the standard $\Lambda$CDM model. This serves as our "Control Group" to ensure the tensor engine and loss functions are bug-free.

*   **Task 1.1: Environment & Project Scaffolding**
    *   Initialize `pyproject.toml` with the JAX/Equinox stack.
    *   Set up the directory structure as defined in Section 3.2.
*   **Task 1.2: Mock Data Generation**
    *   Create a script `scripts/generate_mock_flrw.py` to produce a "Perfect FLRW" universe.
    *   Generate a dataset of $d_L(z)$ values for $z \in [0, 2]$ using standard $\Lambda$CDM parameters ($\Omega_m=0.3, \Omega_\Lambda=0.7, H_0=70$).
*   **Task 1.3: Training the Control Model**
    *   Initialize the 4D Metric PINN but enforce **Isotropy Constraints** ($g_{xx} = g_{yy} = g_{zz}$ and all off-diagonals = 0).
    *   Physics Loss: Include a non-zero $\Lambda$ term to match the mock data: $\mathcal{L}_{phys} = \|G_{\mu\nu} + \Lambda g_{\mu\nu} - 8\pi G T_{\mu\nu}\|^2$.
*   **Task 1.4: Validation & Verification (V&V)**
    *   **Scale Factor Recovery:** Extract $a(t) = \sqrt{g_{xx}(t, 0)}$ from the trained network. It must match the analytic FLRW scale factor within 0.1%.
    *   **Residual Check:** After training, the $G_{\mu\nu}$ components must satisfy the 1st and 2nd Friedmann equations exactly.
    *   **Metric Signature:** Verify via `jax.numpy.linalg.eigvals` that the signature remains $(-, +, +, +)$ at all points in the training domain.

### Phase 2: The 4D Tensor Calculus Engine (The Physics Core)
... (renumbered from here) ...

**Goal:** Construct a robust library to compute $G_{\mu\nu}$ from any network $g_{\mu\nu}(t, x, y, z)$.

*   **Task 2.1: The Metric Wrapper (`src/core/metric.py`)**
    *   Implement `MetricTensor(equinox.Module)`:
        *   Input: $(t, x, y, z)$. Output: 10 scalars representing the upper triangle of $g_{\mu\nu}$.
        *   Hard-code the signature: $g_{00} = -\exp(out_0)$, $g_{ii} = \exp(out_i)$, off-diagonals = $out_j$.
*   **Task 2.2: Automated Tensor Algebra (`src/core/tensors.py`)**
    *   **Step A: Jacobian of the Metric.** Use `jax.jacfwd` to get $\partial_\gamma g_{\mu\nu}$.
    *   **Step B: Inverse Metric.** Use `jax.lax.linalg.inv` (crucial for Christoffel symbols).
    *   **Step C: Christoffel Symbols.** Implement $\Gamma^\alpha_{\mu\nu} = \frac{1}{2} g^{\alpha\sigma}(\partial_\mu g_{\nu\sigma} + \partial_\nu g_{\mu\sigma} - \partial_\sigma g_{\mu\nu})$.
    *   **Step D: Ricci Tensor.** Implement $R_{\mu\nu} = \partial_\rho \Gamma^\rho_{\mu\nu} - \partial_\nu \Gamma^\rho_{\mu\rho} + \Gamma^\rho_{\mu\nu}\Gamma^\sigma_{\rho\sigma} - \Gamma^\sigma_{\mu\rho}\Gamma^\rho_{\nu\sigma}$.
        *   *Note:* This requires the Hessian of the metric. Use `jax.jacfwd(jax.jacrev(model))` for maximum stability.
*   **Task 2.3: Verification (The Schwarzschild Vacuum Test)**
    *   Create `tests/unit/test_tensors.py`.
    *   Define an analytic function for the Schwarzschild metric: $g_{00} = -(1-2M/r)$, $g_{rr} = (1-2M/r)^{-1}$, etc.
    *   Pass this function into the tensor engine.
    *   **Success Criterion:** All components of $R_{\mu\nu}$ must be zero to within $10^{-5}$ tolerance.

### Phase 3: Differentiable Geodesic Ray-Tracer (The Observational Link)
**Goal:** Map the learned metric to the Luminosity Distance $d_L(z)$ using `Diffrax`.

*   **Task 3.1: The Null Geodesic ODE (`src/physics/geodesics.py`)**
    *   Define the system of first-order ODEs:
        1.  $\frac{d x^\mu}{d\lambda} = k^\mu$
        2.  $\frac{d k^\mu}{d\lambda} = -\Gamma^\mu_{\alpha\beta} k^\alpha k^\beta$
    *   Constraint: Light follows null paths ($g_{\mu\nu} k^\mu k^\beta = 0$).
*   **Task 3.2: Differentiable Integration**
    *   Use `diffrax.diffeqsolve` with `Tsit5()` (Runge-Kutta 5/4).
    *   Implement the observer-to-source integration: Start at $t=now, \vec{x}=0$, integrate backwards in affine parameter $\lambda$ until the desired redshift $z$ is reached.
    *   Calculate $d_L(z) = (1+z) r_{prop}$.
*   **Task 3.3: Verification (FLRW Distance Match)**
    *   Set the metric network to a frozen FLRW state.
    *   Run the ray-tracer to compute $d_L(z)$ for $z \in [0, 1.5]$.
    *   **Success Criterion:** $d_L(z)$ must match the analytic FLRW integral $\int dz/H(z)$ within 1%.

### Phase 4: Full Bianchi Training & Analysis (The Discovery Phase)
**Goal:** Train the full 4D PINN on the Pantheon+ Supernova dataset and analyze for shear.

*   **Task 4.1: Data Ingestion (`src/training/data.py`)**
    *   Load `Pantheon+` Supernovae: $(z_{cmb}, \mu_{obs}, \sigma_\mu)$.
    *   Implement coordinate normalization: Map $z \in [0, 2]$ to $t \in [-1, 1]$.
*   **Task 4.2: The Global Loss Optimizer**
    *   Initialize `Optax.adam(learning_rate=1e-4)`.
    *   **The Training Loop:**
        1.  Sample $10,000$ random points $(t,x,y,z)$ for $\mathcal{L}_{Physics}$.
        2.  Compute $G_{\mu\nu}$ at these points. Minimize $\|G_{\mu\nu}\|^2$ (assuming vacuum/dust).
        3.  Compute $d_L(z)$ via Task 3.2 for the Supernova redshifts.
        4.  Minimize $\chi^2 = \sum (\mu_{pred} - \mu_{obs})^2 / \sigma_\mu^2$.
*   **Task 4.3: Deep Metric Extraction & Shear Analysis**
    *   **Goal:** Quantify the physical deviations from FLRW by extracting the shear tensor and expansion rates.
    *   **Implementation:**
        *   Compute the **Shear Tensor** $\sigma_{\mu\nu}$ using the first derivatives of the metric.
        *   Calculate the scalar shear $\sigma^2 = \frac{1}{2}\sigma_{\mu\nu}\sigma^{\mu\nu}$.
        *   Extract the **Hubble Tensor** $H_{ij}$ and compare expansion rates along the x, y, and z axes.
        *   Generate the final **Hubble Diagram** showing the Pantheon+ fit, CC residuals, and shear evolution.
    *   **Success Criterion:** Identification of the primary expansion axis and its alignment with known cosmological structures (e.g., CMB dipole).
*   **Task 4.4: Out-of-Sample Validation (Cosmic Chronometers)**
    *   **Goal:** Verify the PINN-learned expansion history against independent, non-distance-ladder measurements.
    *   **Implementation:**
        *   Ingest the 32-point **Cosmic Chronometers** $H(z)$ dataset.
        *   Extract $H(z)$ directly from the learned metric using $H(z) = -\frac{1}{1+z}\frac{dz}{dt}$.
        *   Compute the residuals and $\chi^2$ against the CC data without further training.
    *   **Success Criterion:** The PINN model must achieve a reduced $\chi^2 \approx 1$ against the independent CC dataset.
*   **Task 4.5: Model Selection & Final Synthesis**
    *   **Goal:** Statistically determine if the inhomogeneous model is superior to $\Lambda$CDM.
    *   **Implementation:**
        *   Compute the **AIC (Akaike Information Criterion)** and **BIC (Bayesian Information Criterion)** for both the PINN and a standard FLRW fit.
    *   **Success Criterion:** Final determination of whether the data statistically justifies a "lumpy" universe over a smooth one.

---

## 6. Testing & Validation
*   **Unit Tests:** Verify $g_{\mu\nu}$ symmetry and signature.
*   **Symmetry Tests:** Check if the learned metric respects the requested Bianchi Type I spatial symmetries.
*   **Performance:** Benchmark training speed. The RTX 5070 must handle $\sim 10^5$ collocation points per minute.
 minute.
