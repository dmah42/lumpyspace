# Technical Design Document: Physics-Informed Neural Network (PINN) for Anisotropic Cosmology

## 1. Executive Summary

This document outlines the architecture, mathematical foundation, and implementation strategy for a Physics-Informed Neural Network (PINN) designed to reconstruct the spacetime metric $g_{\mu\nu}$ of the universe. 

**The Goal:** To model a universe that relaxes the strict homogeneity and isotropy assumptions of the Friedmann-Lemaître-Robertson-Walker (FLRW) metric. We aim to show that by allowing for **anisotropy (shear)** and **local inhomogeneity**, we can fit cutting-edge cosmological data (like the late-time $H_0$ measurements from Supernovae) without relying on ad-hoc "fudge factors" like Dark Energy ($\Lambda$).

**The Approach:** We will utilize a PINN where the neural network acts as a universal function approximator for the metric components $g_{\mu\nu}(t, x, y, z)$. The network is trained not just to fit observational data, but is constrained by the Einstein Field Equations (EFE) serving as a physical loss function. We will specifically lean into **Bianchi Class I/VII** models as our architectural baseline.

**Hardware Target:** Local execution on a single NVIDIA RTX 5070 GPU.

---

## 2. Theoretical Foundation & The "Why"

### 2.1 Why Abandon FLRW?
The FLRW metric assumes the universe is a perfectly smooth fluid. This "Mean Field" approximation fails to account for the non-linear clustering of matter (the Cosmic Web). This failure manifests as the **Hubble Tension** (early vs. late universe expansion rates disagreeing at $5\sigma$) and anomalies in the CMB (e.g., the "Axis of Evil").

### 2.2 The Target Metric: Perturbed Bianchi Class I
Instead of FLRW, the network will learn a metric that is fundamentally anisotropic. We select **Bianchi Type I** as our base because it is the simplest generalization of flat FLRW, allowing for different expansion rates along the $x, y,$ and $z$ axes (introducing **shear**, $\sigma$).

We further allow for local spatial variations (inhomogeneity) to capture Buchert-style backreaction. The line element we are learning is effectively:

$$ds^2 = -dt^2 + g_{xx}(t,\vec{x})dx^2 + g_{yy}(t,\vec{x})dy^2 + g_{zz}(t,\vec{x})dz^2 + g_{ij}(t,\vec{x})dx^i dx^j$$

*Why this metric?* It allows the neural network to naturally discover if shear ($\sigma \neq 0$) or local expansion variance ($\mathcal{Q}_D \neq 0$) can replace $\Lambda$ to fit the distance-redshift data.

---

## 3. Machine Learning Architecture (The PINN)

We will use **JAX** combined with **Equinox** or **Flax**. 
*Why JAX?* PINNs require calculating second-order derivatives of the network output with respect to its inputs to construct the Einstein Tensor ($G_{\mu\nu}$). JAX's forward-mode and reverse-mode automatic differentiation (`jax.grad`, `jax.hessian`) are significantly faster and more memory-efficient for high-order PDEs than PyTorch's computational graph, which is crucial for fitting this onto an RTX 5070.

### 3.1 Network Topology
*   **Inputs:** 4-dimensional spacetime coordinates $(t, x, y, z)$. To handle the vast scales of cosmology, inputs must be normalized (e.g., $t$ in units of Hubble time, spatial coords in Mpc mapped to $[-1, 1]$).
*   **Outputs:** 10 independent components of the symmetric $4\times4$ metric tensor $g_{\mu\nu}$.
*   **Architecture:** A Multi-Layer Perceptron (MLP) with **Fourier Feature Mappings** (or Positional Encoding) at the input layer.
    *   *Why Fourier Features?* Standard MLPs suffer from "spectral bias"—they struggle to learn high-frequency functions. The cosmic web is highly non-linear and "lumpy" (high frequency). Encoding $(t,x,y,z)$ into sine/cosine features allows the network to resolve local voids and filaments.
*   **Activation Function:** `Tanh` or `SiREN` (Sine activations).
    *   *Why SiREN?* The Einstein equations require smooth, continuous second derivatives. `ReLU` has a zero second derivative, making it useless for PINNs solving PDEs. `SiREN` preserves gradients perfectly across layers.

### 3.2 Imposing Constraints via Network Design
To ensure the metric has the correct physical signature $(-, +, +, +)$, the output layer must enforce positivity on the spatial diagonal components and strict negativity on the time component:
*   $g_{00} = - \exp(\text{NetworkOutput}_{0})$
*   $g_{ii} = \exp(\text{NetworkOutput}_{i})$
*   *Why?* This prevents the network from exploring unphysical regions of the parameter space where time becomes space-like, which would cause the loss function to explode.

---

## 4. The Loss Function (The Cosmic Optimizer)

The PINN loss function is a weighted sum of three terms:
$$\mathcal{L} = w_{Data} \mathcal{L}_{Data} + w_{Physics} \mathcal{L}_{Physics} + w_{Prior} \mathcal{L}_{Prior}$$

### 4.1 The Physics Loss ($\mathcal{L}_{Physics}$)
This forces the learned metric to obey General Relativity. We compute the Einstein Tensor $G_{\mu\nu}$ directly from the network outputs using automatic differentiation.
$$\mathcal{L}_{Physics} = \frac{1}{N_{colloc}} \sum_{k=1}^{N_{colloc}} \left\| G_{\mu\nu}(x_k) + \Lambda g_{\mu\nu}(x_k) - 8\pi G T_{\mu\nu}(x_k) \right\|^2$$
*   *Implementation Note:* We will set $\Lambda = 0$ as a hard constraint to see if the network can fit the data *without* Dark Energy.
*   $T_{\mu\nu}$ will be modeled as a pressureless dust (Dark Matter + Baryons).
*   $N_{colloc}$ are "collocation points" randomly sampled across the 4D spacetime domain.

### 4.2 The Data Loss ($\mathcal{L}_{Data}$)
This grounds the model in reality. Our primary target is the Type Ia Supernovae distance modulus ($\mu$).
$$\mathcal{L}_{Data} = \frac{1}{N_{obs}} \sum_{i=1}^{N_{obs}} \left( \mu_{predicted}(z_i) - \mu_{observed}(z_i) \right)^2$$
*   *The Hard Part:* To get $\mu_{predicted}$, we must integrate null geodesics ($ds^2 = 0$) from the observer ($t=now, \vec{x}=0$) to the supernova through our learned, lumpy metric. This requires a differential equation solver integrated *inside* the loss function. 
*   *Why?* We cannot assume the simple FLRW distance-redshift relation $d_L(z)$. The light must travel through the learned anisotropic shear.

### 4.3 The Prior/Regularization Loss ($\mathcal{L}_{Prior}$)
To prevent the model from overfitting the local universe (e.g., creating wild, unphysical wormholes to explain a single outlier supernova), we enforce an asymptotic prior.
$$\mathcal{L}_{Prior} = \lambda \int \| g_{\mu\nu} - g_{\mu\nu}^{FLRW} \|^2 \, d^4x \quad \text{for large } z$$
*   *Why?* The CMB proves the early universe was nearly perfectly isotropic. We penalize deviations from FLRW at high redshifts ($z > 10$), allowing anisotropy to grow only at late times (low redshift), matching the growth of structure.

---

## 5. Execution Plan & RTX 5070 Optimization

Building this is complex. An engineer should tackle this in distinct phases:

### Phase 1: The "Spherical Cow" PINN (Weeks 1-2)
*   **Goal:** Build a 1D PINN that learns the FLRW scale factor $a(t)$.
*   **Action:** Input is just $t$. Output is $a(t)$. Data is mock $H(z)$ data. Physics loss is the standard Friedmann equation.
*   **Why:** Validates the JAX autodiff pipeline and training loop before introducing spatial tensors.

### Phase 2: Autodiff Tensor Calculus Pipeline (Weeks 3-4)
*   **Goal:** Compute $G_{\mu\nu}$ dynamically.
*   **Action:** Write a JAX library that takes the neural network function $f(t,x,y,z) \rightarrow g_{\mu\nu}$ and automatically computes the Christoffel symbols, Riemann tensor, Ricci tensor, and finally the Einstein tensor.
*   **Why:** Hard-coding these derivatives is impossible. We must rely on `jax.jacfwd` and `jax.jacrev`.

### Phase 3: The Geodesic Ray-Tracer (Weeks 5-6)
*   **Goal:** Map $g_{\mu\nu}$ to observables.
*   **Action:** Implement a differentiable ODE solver (e.g., using `Diffrax` in JAX) to calculate the path of light rays from $z=0$ to $z=1.5$.
*   **Why:** Without this, we cannot compare our metric to Supernova data.

### Phase 4: Full Bianchi Integration & Training (Weeks 7-8)
*   **Action:** Train the full model on the Pantheon+ Supernova dataset.
*   **RTX 5070 Constraints:** The 5070 has excellent compute but limited VRAM (likely 12GB).
    *   *Batch Size:* We must use small batch sizes for the observational data, but can sample millions of collocation points for $\mathcal{L}_{Physics}$ using `jax.vmap`.
    *   *Mixed Precision:* Use `bfloat16` for the network weights, but *must* cast to `float32` or `float64` before calculating the Einstein tensor. GR equations are highly sensitive to numerical instability; $16$-bit floats will cause the Christoffel symbols to produce NaNs.

---

## 6. Success Metrics & Analysis

Once the model converges, we will freeze the weights and extract the physical insights:
1.  **Extract the Shear ($\sigma$):** Analyze the off-diagonal spatial components of the learned metric. Does the network discover a preferred direction of expansion that correlates with the CMB Axis of Evil?
2.  **Calculate Local $H_0$:** Compute the expansion rate at $\vec{x}=0$ vs. large distances. Does the model naturally produce a local "Hubble Bubble" to resolve the $73$ vs $67$ km/s/Mpc tension?
3.  **Evaluate $\Lambda$:** If the model fits the Pantheon+ data with high accuracy while keeping the explicit $\Lambda = 0$ in the physics loss, we have successfully demonstrated that anisotropic geometry can replace Dark Energy.