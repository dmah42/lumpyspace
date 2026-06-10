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

### 4.3 The Early-Universe Isotropy and Expansion Prior (The CMB Constraint)
**The Tension:** This project aims to "relax strict homogeneity and isotropy assumptions" to test if local structure can mimic Dark Energy. However, without observational data in the deep past, the neural metric tends to drift into unphysical regimes (e.g., contracting pasts with high shear) because the matter density is too low to dynamically drive expansion. 

**The Resolution (The Sachs-Wolfe Effect):**
We must enforce physical constraints in the early universe ($t \le -3.0$, corresponding to $z \ge 10$) that are consistent with Cosmic Microwave Background (CMB) observations at recombination ($z \approx 1100$). The CMB temperature map is astonishingly uniform, with fluctuations on the order of $\Delta T/T \approx 10^{-5}$. According to the Sachs-Wolfe effect on large scales ($\Delta T/T \approx \frac{1}{3} \Phi/c^2$), these temperature fluctuations are directly proportional to the primordial gravitational potential $\Phi$. This proves with mathematical certainty that at $z \approx 1100$, the universe had almost no "lumps" of density or massive spatial curvature gradients. Furthermore, any significant anisotropic expansion (shear) would induce massive quadrupole temperature fluctuations. 

Because shear mathematically grows as $a^{-6}$ when integrating backwards in time, any non-zero shear today must be vanishingly small in the early universe, or it would explode and blatantly violate CMB observations. Furthermore, because our model broke Bianchi Type I symmetry by developing a spatial curvature dipole, shear is a 4D field $\sigma^2(t, x, y, z)$.

**The Implementation (Phase 4 Spec):**
To ground the metric, we enforce a localized prior as a strict boundary condition in the deep past ($t \in [-4.0, -3.990]$). Because this boundary slice is incredibly narrow, we will explicitly sample it using a smaller batch of `N_cmb = 50` points per step.

At each sampled coordinate in this narrow slice, we must compute the Extrinsic Curvature ($K_{ij}$) and the spatial metric ($g_{ij}$):
1. Extract the $3 \times 3$ spatial metric: $g_{ij} = g[1:4, 1:4]$.
2. Compute the spatial inverse: $g^{ij} = \text{jnp.linalg.inv}(g_{ij})$.
3. Extract the lapse function: $N = \sqrt{-g_{00}}$.
4. Compute the time derivative of the spatial metric: $\dot{g}_{ij} = \partial_t g_{ij}$ (extracted from `jax.jacfwd(metric)`).
5. Calculate the Extrinsic Curvature: $K_{ij} = -\frac{1}{2N} \dot{g}_{ij}$.

With these geometric primitives, we enforce three Augmented Lagrangian penalties, allowing for a realistic $\epsilon = 10^{-5}$ tolerance on the isotropy and homogeneity to prevent over-constraining the model:

1. **The Expansion Prior ($H_{\text{mean}}$)**
   - **Math**: The mean expansion rate is the trace of the extrinsic curvature. $H_{\text{mean}} = -\frac{1}{3} \text{Tr}(K^i_j) = -\frac{1}{3} (g^{ik} K_{kj})$.
   - **Penalty**: $\mathcal{L}_{\text{expand}} = \max(0, -H_{\text{mean}})^2$. We square the ReLU to ensure $C^1$ continuity for the gradients.

2. **The Isotropy Prior (Shear $\sigma^2$)**
   - **Math**: The traceless shear tensor is $\sigma_{ij} = K_{ij} - \frac{1}{3} \text{Tr}(K) g_{ij}$. 
   - **Scalar Formulation**: The shear scalar is $\sigma^2 = \frac{1}{2} \sigma_{ij} \sigma^{ij} = \frac{1}{2} (g^{ia} g^{jb} \sigma_{ij} \sigma_{ab})$.
   - **Penalty**: $\mathcal{L}_{\text{shear}} = \max(0, \sigma^2 - 10^{-5})^2$.

3. **Spatial Homogeneity (Curvature Gradients)**
   - **Math**: We penalize the spatial gradients of all metric components. Let $\partial_k g_{\mu\nu}$ be the spatial Jacobian (derivatives with respect to $x, y, z$).
   - **Penalty**: $\mathcal{L}_{\text{spatial}} = \max(0, \sum_{k=1}^3 \sum_{\mu,\nu=0}^3 (\partial_k g_{\mu\nu})^2 - 10^{-5})^2.

**Augmented Lagrangian Architecture:**
Because these three penalties represent distinct physical constraints, summing them under a single $\lambda$ multiplier would cause gradient imbalance (e.g., the network trades shear violation for expansion violation).
*   **Independent Tracking**: The model state must be updated to track three separate `AdaptivePenaltyState` schedulers and three independent $\lambda$ multipliers (`lambda_cmb_expand`, `lambda_cmb_shear`, `lambda_cmb_spatial`).
*   **Initialization**: Initialize all three penalties with a starting weight $w=1.0$ and $\lambda=0.0$.

**Stitching the Boundary (EFE Overlap):**
Applying these penalties at $t \in [-4.0, -3.990]$ acts as a discrete boundary snapshot. To connect this boundary to the late universe, the Einstein Field Equations (EFE) must be stitched seamlessly to it. Therefore, the continuous EFE sampling domain (`t_inactive`) must be extended down to $t=-3.990$. Because the EFE physical domain touches the CMB boundary, the differential equations will "grab" the expanding, isotropic boundary condition and mathematically propagate it forward through the unobserved region all the way into the active Supernova region.

*Note on Stiffness*: Extending the EFE sampling down into the $-3.990$ region will introduce incredibly massive density gradients ($\rho(z) \propto (1+z)^3$). Testing the EFE extension (moving the `minval` from $-3.9$ to $-3.990$) to ensure the Spatial Curriculum can handle the mathematical stiffness without `NaN` explosions is a critical prerequisite before activating the CMB boundary loss.

---

## 5. Phased Implementation Roadmap

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

*   **Task 4.1: Data Ingestion (`src/training/data.py`) [COMPLETED]**
    *   Load `Pantheon+` Supernovae: $(z_{cmb}, \mu_{obs}, \sigma_\mu)$.
    *   Implement coordinate normalization: Map $z \in [0, 2]$ to $t \in [-1, 1]$.
*   **Task 4.2: The Global Loss Optimizer [COMPLETED]**
    *   **Learning Rate Scheduler:** Implement an `optax.warmup_cosine_decay_schedule` to navigate the rugged Tabula Rasa loss landscape. The schedule must be parameterized (`warmup_steps`, `decay_steps`) to provide a sharp "kick" out of local minima followed by a smooth descent.
    *   **Optimizer:** Initialize `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))`.
    *   **The Training Loop:**
        1.  Sample $10,000$ random points $(t,x,y,z)$ for $\mathcal{L}_{Physics}$.
        2.  Compute $G_{\mu\nu}$ at these points. Minimize $\|G_{\mu\nu}\|^2$ (assuming vacuum/dust).
        3.  Compute $d_L(z)$ via Task 3.2 for the Supernova redshifts.
        4.  Minimize $\chi^2 = \sum (\mu_{pred} - \mu_{obs})^2 / \sigma_\mu^2$.

*   **Task 4.3: Automated Hyperparameter Optimization (Gradient Balancing) [COMPLETED]**
    *   **Goal:** Automate the tuning of the multi-probe loss weights ($w_{sn}$, $w_{bao}$) to prevent the physics loss ($G_{\mu\nu}=0$) and data losses from overwhelming each other.
    *   **Implementation Options:**
        1.  **Bayesian Optimization (Optuna/Ray Tune):** Wrapping the training loop in an external framework to run dozens of trials, hunting for the optimal static weights that minimize validation loss.
            *   *Tradeoffs:* Highly robust and guaranteed to find stable static weights. However, it is computationally expensive (requires running the full PINN training loop dozens of times) and uses static weights that cannot adapt if the loss landscape changes drastically during training.
        2.  **Self-Adaptive Loss Weights (The PINN Approach):** Making the weights ($w_{sn}$, $w_{bao}$) learnable parameters within the JAX/Equinox model itself. The loss function is modified to simultaneously minimize the residuals while maximizing the weights (e.g., $\mathcal{L} = \sum \frac{1}{2 w_i^2} \mathcal{L}_i + \log(w_i)$), allowing the optimizer to dynamically balance the gradients at every step.
            *   *Tradeoffs & Failure Mode:* While computationally elegant, this method suffers from the "Lazy Optimizer" problem when applied to highly rigid differential equations like the EFE. If the metric initialization produces massive curvature gradients that are difficult to minimize, the optimizer finds it mathematically cheaper to simply drive $w_i \to \infty$. This zeros out the effective loss $\left( \frac{1}{2 w_i^2} \to 0 \right)$, stalling training completely. In our empirical testing, meticulously hand-tuned static hyperparameter weights proved far superior as they act as unyielding constraints that force the network to actually solve the PDEs.

*   **Task 4.4: Deep Metric Extraction & Shear Analysis [IN PROGRESS]**
    *   **Goal:** Quantify the physical deviations from FLRW by extracting the shear tensor and expansion rates.
    *   **Implementation:**
        *   Compute the **Shear Tensor** $\sigma_{\mu\nu}$ using the first derivatives of the metric.
        *   Calculate the scalar shear $\sigma^2 = \frac{1}{2}\sigma_{\mu\nu}\sigma^{\mu\nu}$.
        *   Extract the **Hubble Tensor** $H_{ij}$ and compare expansion rates along the x, y, and z axes.
        *   Generate the final **Hubble Diagram** showing the Pantheon+ fit, CC residuals, and shear evolution.
    *   **Success Criterion:** Identification of the primary expansion axis and its alignment with known cosmological structures (e.g., CMB dipole).

*   **Task 4.5: Sparse Observational Datasets (Cosmic Chronometers & BAO) [REJECTED]**
    *   **Goal:** Integrate additional temporal and distance measurements to further constrain the metric beyond Supernova data.
    *   **Implementation:**
        *   *Cosmic Chronometers (CC):* Utilizing directional $dt/dz$ measurements from passively evolving galaxies.
        *   *Baryon Acoustic Oscillations (BAO):* Utilizing 3D standard ruler data derived from early universe plasma physics.
    *   **Reason for Rejection:** Both the CC and BAO datasets are currently too sparse (e.g., ~30 high-quality CC data points) to provide a robust training signal. Incorporating them would risk overfitting the highly flexible 10-dimensional metric network to small-sample noise rather than generalized physical geometry.

*   **Task 4.6: Advanced Probes & Geometries [COMPLETED]**
    *   *CMB Temperature Power Spectrum:* Mapping our 4D metric fluctuations to the CMB angular power spectrum to ensure the late-time shear does not violate the high-redshift isotropy of the CMB ($z \approx 1100$).
    *   *Inhomogeneous and Anisotropic Spacetimes (Szekeres):* Relaxing the
        Bianchi Type I symmetry to allow for full spatial inhomogeneity while
        preserving directional expansion. We can model Szekeres spacetimes
        analytically to test if localized curvature gradients and non-zero
        shear can simultaneously explain both the Hubble Tension and the
        illusion of Dark Energy.

*   **Task 4.7: Late-Time Isotropy Penalty (Galaxy Surveys) [FUTURE]**
    *   **Goal:** Align the model's geometry with macroscopic observations of the local universe today ($t=0$).
    *   **Implementation:** Observational data from all-sky local galaxy surveys (e.g., [2MASS](https://www.ipac.caltech.edu/2mass/)) indicates that the macroscopic distribution of matter and expansion is isotropic to within roughly 1% to 2%. To match this observation, we will implement a soft Augmented Lagrangian penalty at $t=0$. The penalty will only trigger if the variance between the directional Hubble parameters ($H_x, H_y, H_z$) exceeds a 2% threshold. This anchors the metric to the physical reality of the local universe while mathematically allowing for small, physically viable anisotropic solutions.

*   **Task 4.8: Model Selection & Final Synthesis [PENDING]**
    *   **Goal:** Statistically determine if the inhomogeneous model is superior to $\Lambda$CDM.
    *   **Implementation:**
        *   Compute the **AIC (Akaike Information Criterion)** and **BIC (Bayesian Information Criterion)** for both the PINN and a standard FLRW fit.
    *   **Success Criterion:** Final determination of whether the data statistically justifies a "lumpy" universe over a smooth one.

---

## 5. Advanced Training Dynamics: Batching & Gradient Balancing

To transition the PINN training loop from a prototype to a production-grade optimization engine, we must address the computational complexity of the data loss and the inherent competition between the physics and observational objectives.

### 5.1 Supernova Batching Design
Rather than evaluating all $1,701$ supernovae in the Pantheon+ dataset at each training step—which requires solving $1,701$ separate null geodesic ODEs backwards through time—we introduce stochastic batching of the observational data.

*   **JAX-Compatible Sampling:** We define a static batch size $B$ (e.g., $B = 128$). At each step, we use `jax.random.choice` to sample $B$ indices from the dataset. By keeping the batch size constant, JAX's compilation engine (XLA) avoids recompilation.
*   **The Stochastic Gradient Advantage:** Evaluating a subset of geodesics introduces stochastic noise into the observational loss gradient $\nabla_\theta \mathcal{L}_{Data}$. While this noise requires a slightly smaller base learning rate, it acts as a regularizer. The resulting "gradient thermal bath" helps the network parameters jump out of the shallow local minima and coordinate traps that characterize the unconstrained 10-dimensional metric landscape.
*   **Implementation Details:**
    *   `batch_z, batch_mu, batch_err = select_batch(sn_data, indices)`
    *   The loss is normalized by the batch size $B$ to maintain consistent gradient scaling across different choices of $B$.

### 5.2 Hyperparameter Tuning & Gradient Balancing
Physics-Informed Neural Networks often fail when the gradients of the physics loss ($\mathcal{L}_{Physics}$) and the data loss ($\mathcal{L}_{Data}$) have vastly different magnitudes. If one dominates, the optimizer will satisfy it at the absolute expense of the other.

#### 5.2.1 Biasing Towards Physics (The Cosmological Constraint)
Because a metric that fits the supernova data but violates Einstein's Field Equations is physically meaningless, we must bias the optimization toward satisfying General Relativity. However, if the physics loss weight $w_{Physics}$ is set too high, the model quickly collapses into the trivial vacuum Minkowski metric (which has exactly zero curvature and zero physics loss) and refuses to ever move toward fitting the expansion of the universe.
To prevent this, we propose two methods:

#### 5.2.2 Option 1: Dynamic GradNorm (Gradient Norm Balancing)
We dynamically adjust the loss weights $w_i(t)$ at each training step $t$ to ensure the gradients of the physics and data losses remain in a physical proportion.
*   **The Algorithm:**
    1.  At step $t$, compute the gradients of the model's final layer parameters with respect to each individual loss component: $G_{phys}(t) = \|\nabla_{\theta_{last}} (w_{phys} \mathcal{L}_{phys})\|_2$ and $G_{data}(t) = \|\nabla_{\theta_{last}} (w_{data} \mathcal{L}_{data})\|_2$.
    2.  Define a target ratio $\alpha$ (e.g., $\alpha = 10.0$ to bias the optimizer towards physics).
    3.  Compute the desired weight updates to bring the gradient ratio close to $\alpha$:
        $$w_{phys}(t+1) = (1 - \beta) w_{phys}(t) + \beta \left( \alpha \frac{\bar{G}(t)}{G_{phys}(t)} \right)$$
        where $\bar{G}(t)$ is the mean gradient norm, and $\beta$ is a momentum hyperparameter (e.g., $0.1$).
*   **Tradeoffs:** Automatically prevents either loss from "drowning out" the other, stabilizing training across the entire 10,000-step cycle. However, computing separate gradients for each loss component adds slight backpropagation overhead.

#### 5.2.3 Option 2: Augmented Lagrangian (Hard Boundary Constraints)
We reformulate the training as a constrained optimization problem, treating the Einstein Field Equations as a hard constraint:
$$\min_{\theta} \mathcal{L}_{Data}(\theta) \quad \text{subject to} \quad \mathcal{L}_{Physics}(\theta) \le \epsilon$$
We optimize the Augmented Lagrangian:
$$\mathcal{L}_{Aug}(\theta, \lambda) = \mathcal{L}_{Data}(\theta) + \lambda \mathcal{L}_{Physics}(\theta) + \frac{\mu}{2} \mathcal{L}_{Physics}(\theta)^2$$
Where:
*   $\lambda$ is the Lagrange multiplier, updated after every epoch/batch step: $\lambda \leftarrow \lambda + \mu \mathcal{L}_{Physics}(\theta)$.
*   $\mu$ is the penalty parameter, which increases over training to penalize constraint violations more severely.
*   **Tradeoffs:** Guarantees that the final converged metric strictly satisfies General Relativity. However, it requires careful tuning of $\mu$'s growth rate to prevent numerical instability.

#### 5.2.4 Adaptive Penalty Scheduling
To automate the tuning of the Augmented Lagrangian penalty parameter $\mu$ (e.g., `w_wec`), we implement an Adaptive Penalty Schedule in the outer Python training loop. This prevents the constraint from stalling in the late stages of training when the violation becomes extremely small.
*   **The Algorithm:**
    1. Define a check interval $N_{check}$ (e.g., every 500 steps).
    2. At step $t = k \cdot N_{check}$, compare the current exponential moving average of the violation $\mathcal{L}_{k}$ against the violation at the previous check $\mathcal{L}_{k-1}$.
    3. If the violation has not decreased by a sufficient margin (e.g., $\mathcal{L}_{k} > \tau \mathcal{L}_{k-1}$, where $\tau = 0.9$):
        *   Multiply the penalty parameter by a growth factor: $\mu \leftarrow \gamma \mu$ (e.g., $\gamma = 1.5$ or $2.0$).
    4. Cap $\mu$ at a massive maximum value (e.g., $10^5$) to prevent `NaN` explosions.
*   **Implementation:** Because $\mu$ (e.g., `w_wec_val`) is passed to the JIT-compiled `step` function as a dynamic `jnp.ndarray`, we can seamlessly multiply it by $\gamma$ in the outer Python loop without triggering recompilation. This system will natively scale to handle multiple constraints (WEC, CMB Expansion, CMB Isotropy) independently.

### 5.3 Parameter Sensitivity & Gradient Boosting (The Scalar Multiplier)
A major bottleneck when optimizing deep network parameters alongside single physical scalars (such as the trainable matter density $\kappa\rho_0$) is gradient scale imbalance.

*   **The Problem:** The network contains thousands of weights that determine the complex curvature profiles, making the loss highly sensitive to their changes. These weights dominate the global gradient vector. When `optax.clip_by_global_norm(1.0)` is applied, it divides the entire gradient vector by the global norm. Consequently, the gradient of the single scalar $\kappa\rho_0$ is scaled down to near-zero (e.g. $10^{-4}$), causing it to move extremely slowly over training steps.
*   **The Solution (The "Fast Track" Boost):** Rather than raising the global learning rate (which destabilizes the network and causes NaN gradient explosions), we apply a dedicated gradient multiplier to $\kappa\rho_0$ in the training step:
    $$\nabla_{\kappa\rho_0} \mathcal{L}_{Total} \leftarrow \eta \cdot \nabla_{\kappa\rho_0} \mathcal{L}_{Total}$$
    where $\eta \in [10.0, 100.0]$ is a boosting factor (e.g. $\eta = 50.0$).
*   **Implementation:** Inside the training JIT step, after computing `grads`, we use `equinox.tree_at` to scale `grads.kappa_rho_0` before passing it to the optimizer:
    ```python
    boosted_grad = grads.kappa_rho_0 * 50.0
    grads = eqx.tree_at(lambda m: m.kappa_rho_0, grads, boosted_grad)
    ```
*   **Physical Justification:** The matter density represents a global physical constant that should adjust rapidly to balance the EFE, while the metric network weights must adjust slowly and smoothly to avoid introducing singular coordinate ripples.

### 5.4 Trainable Matter Density Parameterization & Dynamic Floor
To ensure the matter density parameter $\kappa\rho_0$ remains physically
consistent without freezing optimization gradients near constraints, we
implement a shifted softplus parameterization with a dynamically scaled floor.

*   **The Problem (Gradient Freezing):** In early versions, a hard baryonic
    floor was enforced via `jnp.maximum(kappa_rho_0, floor)`. However, if the
    parameter dropped below the constraint, the gradient of the loss with
    respect to the parameter became exactly zero, permanently trapping the
    parameter in a coordinate lock.
*   **The Solution (Softplus Parameterization):** We map a raw unconstrained
    parameter $\theta \in \mathbb{R}$ to the physical density parameter
    $\kappa\rho_0$ via a shifted softplus function:
    $$\kappa\rho_0 = \text{Baryonic Floor} + \text{softplus}(\theta)$$
    Because the derivative of `softplus(x)` is `sigmoid(x)` (which is non-zero
    everywhere), the gradient path remains active for all values of $\theta$.
*   **Dynamic Baryonic Floor:** Rather than defining the floor using a static
    approximation of $H_0 \approx 1$ (e.g. $0.05 \times 3 = 0.15$), the floor
    is scaled dynamically by the model's derived expansion rate today
    $H_{mean}(1.0)$:
    $$\text{Baryonic Floor} = 3 \cdot \Omega_b \cdot H_{mean}(1.0)^2$$
    where $\Omega_b = 0.05$ represents the minimum baryonic matter density
    today. This ensures that the matter density bounds are physically
    consistent with the derived cosmology at each step of training.

### 5.5 EFE Coordinate Sampling & Active Redshift Region Prioritization
To prevent the metric network from exploiting unobserved gaps in the coordinate space to build unphysical coordinate singularities (e.g. "gravitational redshift pumps" designed to fake expansion history), we implement a hierarchical sampling scheme for the training domain.

*   **The Problem (Adversarial Singularities):** Under sparse uniform sampling of coordinate time $t$, a narrow coordinate singularity (width $\approx 0.1$) can easily hide in the gaps between points. The network exploits this to generate the required observational redshift in a single violent contraction-expansion bounce. Furthermore, the early universe ($t \in [-4.0, -2.5]$) exhibits extremely steep, sharp geometric warping that will be mathematically aliased if not resolved with high point density.
*   **The Solution (Massive Resolution Bumping & Density Ratio):** We utilize a dense, two-tiered sampling strategy to fully utilize GPU compute headroom while mathematically oversampling the Supernova domain:
    *   **Active Supernova Region:** We pack **800 points** into $t \in [-2.5, 1.0]$. With a span of 3.5, this yields $\approx 228$ points per unit redshift. This heavily anchors the network to the Pantheon+ dataset.
    *   **Deep Past Extension:** We pack **200 points** into $t \in [-4.0, -2.5]$. With a span of 1.5, this yields $\approx 133$ points per unit redshift. This is still a massive absolute resolution bump (preventing deep-past aliasing) but preserves the mathematical priority of the active region.

### 5.6 Non-Linear Coordinate Mappings & Mixed Tensor Formulation
Transitioning from a linear time mapping ($t=-z$) to a non-linear mapping (e.g., $t=5a-4$) introduces severe optimization challenges for the PINN:
*   **Coordinate Variance of the EFE:** The covariant Einstein tensor $G_{\mu\nu}$ scales heavily with the coordinate choice. Under $t=5a-4$, the covariant residual $G_{tt} - T_{tt}$ is artificially amplified at high redshifts compared to $t=-z$. 
*   **The Mixed Tensor Solution:** To decouple the optimization landscape from the arbitrary choice of $t$, we compute the EFE residual using **Mixed Tensors**: $G^\mu_\nu - 8\pi G T^\mu_\nu = 0$. For a dust universe, $T^t_t = -\rho$ exactly, rendering the target perfectly coordinate-invariant.
*   **The Physical Sampling Trap:** Even with mixed tensors, the physical density $\rho$ scales as $a^{-3}$. At $t=-3.5$ ($z=9$), the squared EFE residual is $10^6$ times larger than today. Because the loss is calculated as a `jnp.mean`, increasing the *density* of sampled points in the early universe will cause this $10^6$ magnitude to completely overwhelm the optimizer, destroying late-time metrics (like the WEC penalty). 
*   **The Conclusion:** It is physically impossible to safely extend the sampling domain to high redshifts (e.g., to fully cover the $z \approx 2.3$ Supernova data) without implementing **Dynamic Gradient Balancing** (Task 4.7) or Relative Weighting (e.g., dividing the residual by $a(t)^4$). Without this balancing, the massive physical gradients from the early universe will inevitably drown out the late-time WEC and observational losses.

### 5.7 Spatially Adaptive Weighting (Solving the Coordinate Imbalance)

#### The Problem: Non-Linear Mapping Imbalance
While global parameters and physical $\gamma$ volume scaling correctly normalize
the *physical* density ($a^{-3}$) across the universe, we still face an internal
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

Using the homoscedastic uncertainty formulation *per-point*:

$$\mathcal{L}_{phys} = \frac{1}{N} \sum_{i=1}^N \left( \frac{1}{2} e^{-2 W(t_i)} \cdot \mathcal{L}_{residual}(t_i) + W(t_i) \right)$$

1. **Dynamic Suppression:** If the WEC or EFE residual is massive at a specific
   boundary $t=-3.5$, the optimizer will locally increase $W(-3.5)$ to suppress
   that stiff gradient.
2. **Spatial Curriculum Learning:** By suppressing the hardest coordinate
   domains, the network is free to solve the PDE in the "easy" regions first. As
   the easy regions converge, their gradients drop, and the network can slowly
   decrease $W(t)$ in the hard regions to tackle them sequentially, acting as a
   completely automated spatial curriculum.

#### Architectural Implementation Details
*   **The Network:** A very lightweight Multi-Layer Perceptron (e.g., 2 hidden
    layers of 16 neurons) appended as a submodule within `MetricNN`. 
*   **Input:** Because the network has discovered an *inhomogeneous* cosmology
    (e.g., a spatial curvature dipole), we have categorically broken the Bianchi
    Type I homogeneity assumption. Therefore, the spatial weight network must
    take the full 4D coordinate vector $(t, x, y, z)$ as input to map out a
    complete 4D spatiotemporal attention mask.
*   **Optimization:** The $W(t, x, y, z)$ network parameters are updated
    simultaneously with the main metric weights. Because it is evaluated inside
    the spatial `jax.vmap` batch, it adds virtually zero computational overhead.

---

## 6. Testing & Validation
*   **Unit Tests:** Verify $g_{\mu\nu}$ symmetry and signature.
*   **Symmetry Tests:** Check if the learned metric respects the requested Bianchi Type I spatial symmetries.
*   **Performance:** Benchmark training speed. The RTX 5070 must handle $\sim 10^5$ collocation points per minute.
