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
    *   **Expansion Prior (Symmetry Breaking):** To prevent the unconstrained network from falling into the mathematically valid "contracting universe" local minimum, we apply a massive `jnp.maximum(0, -H_mean)` penalty. Once the network enters the expanding regime ($H > 0$), this penalty drops strictly to zero.
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
    *   **Learning Rate Scheduler:** Implement an `optax.warmup_cosine_decay_schedule` to navigate the rugged Tabula Rasa loss landscape. The schedule must be parameterized (`warmup_steps`, `decay_steps`) to provide a sharp "kick" out of local minima followed by a smooth descent.
    *   **Optimizer:** Initialize `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))`.
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
*   **Task 4.4: Out-of-Sample Validation & The "Hubble Dipole" (Cosmic Chronometers)**
    *   **Goal:** Verify the PINN-learned expansion history against independent, non-distance-ladder measurements, while accounting for the model's extreme spatial anisotropy.
    *   **Implementation (Future Plan):**
        *   *The Anisotropy Problem:* The standard 32-point Cosmic Chronometers dataset assumes isotropy by averaging $H(z)$ across all sky coordinates. Since our Tabula Rasa PINN predicts a universe expanding significantly faster along a preferred axis (mimicking Dark Energy via shear), testing against a 1D $H(z)$ curve would erase the very physics we discovered.
        *   *The 3D Solution:* We must ingest the full CC dataset *including* the Right Ascension (RA) and Declination (Dec) of each galaxy. We will then shoot 3D null geodesics in those exact angular directions to extract the strictly directional expansion rate $H_{dir}(z) = -\frac{1}{1+z}\frac{dz}{d\tau}$ for each specific data point.
    *   **Success Criterion:** If the highly anisotropic model fits the directional CC data better than a standard isotropic $\Lambda$CDM model, it provides strong observational evidence for the "Hubble Dipole" and anisotropic Dark Energy.

*   **Task 4.5: Model Selection & Final Synthesis**
    *   **Goal:** Statistically determine if the inhomogeneous model is superior to $\Lambda$CDM.
    *   **Implementation:**
        *   Compute the **AIC (Akaike Information Criterion)** and **BIC (Bayesian Information Criterion)** for both the PINN and a standard FLRW fit.
    *   **Success Criterion:** Final determination of whether the data statistically justifies a "lumpy" universe over a smooth one.

*   **Task 4.6: Future Directions & Additional Observational Probes**
    *   *Baryon Acoustic Oscillations (BAO):* BAO provides an absolute distance scale (standard ruler) derived from the physics of the early universe plasma. Fitting the PINN against 3D BAO measurements (transverse vs. line-of-sight) would be the ultimate test of anisotropic geometry, as BAO measurements are highly sensitive to directional expansion.
    *   *CMB Temperature Power Spectrum:* Mapping our 4D metric fluctuations to the CMB angular power spectrum to ensure the late-time shear does not violate the high-redshift isotropy of the CMB ($z \approx 1100$).
    *   *Local Density Backreaction (Lemaître-Tolman-Bondi):* Relaxing the Bianchi Type I symmetry to allow for full radial inhomogeneity. We can test if we live inside a massive local void, where negative spatial curvature acts as an effective repulsive force (mimicking $\Lambda$).

*   **Task 4.7: Automated Hyperparameter Optimization (Gradient Balancing)**
    *   **Goal:** Automate the tuning of the multi-probe loss weights ($w_{sn}$, $w_{bao}$) to prevent the physics loss ($G_{\mu\nu}=0$) and data losses from overwhelming each other.
    *   **Implementation Options:**
        1.  **Bayesian Optimization (Optuna/Ray Tune):** Wrapping the training loop in an external framework to run dozens of trials, hunting for the optimal static weights that minimize validation loss.
            *   *Tradeoffs:* Highly robust and guaranteed to find stable static weights. However, it is computationally expensive (requires running the full PINN training loop dozens of times) and uses static weights that cannot adapt if the loss landscape changes drastically during training.
        2.  **Self-Adaptive Loss Weights (The PINN Approach):** Making the weights ($w_{sn}$, $w_{bao}$) learnable parameters within the JAX/Equinox model itself. The loss function is modified to simultaneously minimize the residuals while maximizing the weights (e.g., $\mathcal{L} = \sum \frac{1}{2 w_i^2} \mathcal{L}_i + \log(w_i)$), allowing the optimizer to dynamically balance the gradients at every step.
            *   *Tradeoffs:* Computationally cheap (only requires a single training run) and dynamically adapts to the rugged loss landscape in real-time. However, it can be numerically unstable and requires careful initialization to prevent the optimizer from exploiting the weights to artificially zero out the loss.

---

## 6. Advanced Training Dynamics: Batching & Gradient Balancing

To transition the PINN training loop from a prototype to a production-grade optimization engine, we must address the computational complexity of the data loss and the inherent competition between the physics and observational objectives.

### 6.1 Supernova Batching Design
Rather than evaluating all $1,701$ supernovae in the Pantheon+ dataset at each training step—which requires solving $1,701$ separate null geodesic ODEs backwards through time—we introduce stochastic batching of the observational data.

*   **JAX-Compatible Sampling:** We define a static batch size $B$ (e.g., $B = 128$). At each step, we use `jax.random.choice` to sample $B$ indices from the dataset. By keeping the batch size constant, JAX's compilation engine (XLA) avoids recompilation.
*   **The Stochastic Gradient Advantage:** Evaluating a subset of geodesics introduces stochastic noise into the observational loss gradient $\nabla_\theta \mathcal{L}_{Data}$. While this noise requires a slightly smaller base learning rate, it acts as a regularizer. The resulting "gradient thermal bath" helps the network parameters jump out of the shallow local minima and coordinate traps that characterize the unconstrained 10-dimensional metric landscape.
*   **Implementation Details:**
    *   `batch_z, batch_mu, batch_err = select_batch(sn_data, indices)`
    *   The loss is normalized by the batch size $B$ to maintain consistent gradient scaling across different choices of $B$.

### 6.2 Hyperparameter Tuning & Gradient Balancing
Physics-Informed Neural Networks often fail when the gradients of the physics loss ($\mathcal{L}_{Physics}$) and the data loss ($\mathcal{L}_{Data}$) have vastly different magnitudes. If one dominates, the optimizer will satisfy it at the absolute expense of the other.

#### 6.2.1 Biasing Towards Physics (The Cosmological Constraint)
Because a metric that fits the supernova data but violates Einstein's Field Equations is physically meaningless, we must bias the optimization toward satisfying General Relativity. However, if the physics loss weight $w_{Physics}$ is set too high, the model quickly collapses into the trivial vacuum Minkowski metric (which has exactly zero curvature and zero physics loss) and refuses to ever move toward fitting the expansion of the universe.
To prevent this, we propose two methods:

#### 6.2.2 Option 1: Dynamic GradNorm (Gradient Norm Balancing)
We dynamically adjust the loss weights $w_i(t)$ at each training step $t$ to ensure the gradients of the physics and data losses remain in a physical proportion.
*   **The Algorithm:**
    1.  At step $t$, compute the gradients of the model's final layer parameters with respect to each individual loss component: $G_{phys}(t) = \|\nabla_{\theta_{last}} (w_{phys} \mathcal{L}_{phys})\|_2$ and $G_{data}(t) = \|\nabla_{\theta_{last}} (w_{data} \mathcal{L}_{data})\|_2$.
    2.  Define a target ratio $\alpha$ (e.g., $\alpha = 10.0$ to bias the optimizer towards physics).
    3.  Compute the desired weight updates to bring the gradient ratio close to $\alpha$:
        $$w_{phys}(t+1) = (1 - \beta) w_{phys}(t) + \beta \left( \alpha \frac{\bar{G}(t)}{G_{phys}(t)} \right)$$
        where $\bar{G}(t)$ is the mean gradient norm, and $\beta$ is a momentum hyperparameter (e.g., $0.1$).
*   **Tradeoffs:** Automatically prevents either loss from "drowning out" the other, stabilizing training across the entire 10,000-step cycle. However, computing separate gradients for each loss component adds slight backpropagation overhead.

#### 6.2.3 Option 2: Augmented Lagrangian (Hard Boundary Constraints)
We reformulate the training as a constrained optimization problem, treating the Einstein Field Equations as a hard constraint:
$$\min_{\theta} \mathcal{L}_{Data}(\theta) \quad \text{subject to} \quad \mathcal{L}_{Physics}(\theta) \le \epsilon$$
We optimize the Augmented Lagrangian:
$$\mathcal{L}_{Aug}(\theta, \lambda) = \mathcal{L}_{Data}(\theta) + \lambda \mathcal{L}_{Physics}(\theta) + \frac{\mu}{2} \mathcal{L}_{Physics}(\theta)^2$$
Where:
*   $\lambda$ is the Lagrange multiplier, updated after every epoch/batch step: $\lambda \leftarrow \lambda + \mu \mathcal{L}_{Physics}(\theta)$.
*   $\mu$ is the penalty parameter, which increases over training to penalize constraint violations more severely.
*   **Tradeoffs:** Guarantees that the final converged metric strictly satisfies General Relativity. However, it requires careful tuning of $\mu$'s growth rate to prevent numerical instability.

---

## 7. Testing & Validation
*   **Unit Tests:** Verify $g_{\mu\nu}$ symmetry and signature.
*   **Symmetry Tests:** Check if the learned metric respects the requested Bianchi Type I spatial symmetries.
*   **Performance:** Benchmark training speed. The RTX 5070 must handle $\sim 10^5$ collocation points per minute.
