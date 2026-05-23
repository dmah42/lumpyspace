# Technical Design: Integrating 3D Baryon Acoustic Oscillations (BAO)

## 1. Executive Summary
This document outlines the engineering plan for integrating Baryon Acoustic Oscillations (BAO) into the Physics-Informed Neural Network (PINN). Unlike Supernovae, which measure 1D luminosity distance, BAO natively provides 3D geometry measurements split into **Transverse Distance ($D_M$)** and **Radial Distance ($D_H$)**. This makes it the ultimate dataset to validate whether the "weaponized shear" discovered by the Tabula Rasa PINN is physically viable or simply a mathematical artifact.

## 2. Theoretical Framework & The Sound Horizon
BAO observables are reported as ratios over the comoving sound horizon at the drag epoch ($D_M(z)/r_d$ and $D_H(z)/r_d$). 

**Decision:** We will **hardcode** $r_d$ to the standard Planck value ($r_d \approx 147.09$ Mpc).
*Justification:* The PINN has demonstrated that the spatial shear $\sigma^2$ is exactly zero in the early universe ($t \ll 0$), meaning the model is perfectly isotropic and standard at the time of the CMB. Therefore, relying on the standard FLRW CMB calculation for the primordial sound horizon is theoretically sound. Furthermore, hardcoding an absolute physical ruler prevents the optimizer from cheating the Einstein Field Equations via arbitrary scaling.

## 3. Implementation Plan (Step-by-Step)

### 3.1. Data Pipeline
**1. Create `scripts/generate_bao_data.py`**
*   Write a script to generate a CSV file containing the standard SDSS DR12 consensus BAO measurements (e.g., at $z_{eff} = 0.38, 0.51, 0.61$).
*   The script should output `data/bao_consensus.csv` with columns: `z`, `DM_over_rd`, `DH_over_rd`, and include the covariance matrix elements.

**2. Update `src/training/data.py`**
*   Implement `load_bao_data(file_path="data/bao_consensus.csv") -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]`.
*   It should parse the CSV and return `z_vals`, `dm_obs`, `dh_obs`, and the inverted covariance matrices `cov_inv`.

### 3.2. Physics Engine (`src/physics/geodesics.py`)
**1. Implement `get_bao_distances(metric_fn, z_target)`**
*   *Algorithm:* To account for the model's severe anisotropy, we cannot assume distances are equal in all directions. 
*   Shoot 3 null geodesics backwards from the observer ($t=1.0$) along the principal spatial axes (x, y, z) until they hit $z_{target}$.
*   **Radial Distance ($D_H$):** Extract the local Hubble parameter at the point of intersection. $D_H(z) = c / H(z)$. (In code units, $D_H(z) = 1 / H(z)$).
*   **Transverse Distance ($D_M$):** Compute the comoving transverse distance along the geodesic. For a null ray, $D_M(z) = d_L(z) / (1+z)$.
*   Return the angular average of these distances to compare against the survey consensus data.

### 3.3. Loss Function (`src/training/loss.py`)
**1. Implement `get_bao_loss(metric_fn, z_vals, dm_obs, dh_obs, cov_inv)`**
*   Use `jax.vmap` to vectorize `get_bao_distances` over the BAO redshifts.
*   Construct the residual vector $\vec{\Delta}$ for each redshift: $\Delta = [D_M/r_d - (D_M/r_d)_{obs}, D_H/r_d - (D_H/r_d)_{obs}]$.
*   Compute the $\chi^2$ loss using the inverted covariance matrix: $\chi^2 = \vec{\Delta}^T C^{-1} \vec{\Delta}$.
*   Return the mean $\chi^2$ loss.

### 3.4. Trainer Integration (`src/training/trainer.py` & `src/training/run.py`)
**1. Update `train_model` signature:**
*   Add `data_bao` to the arguments.
*   Add a hyperparameter `w_bao: float = 1.0` (weight for the BAO loss).
*   Extract the Supernova weight into a generic `w_sn: float = 10.0` rather than hardcoding `10.0`.

**2. Update Training Loop:**
*   Call `l_bao = get_bao_loss(...)`.
*   Update the total objective: `total_loss = l_phys + w_sn * l_sn + w_bao * l_bao`.
*   Add `l_bao` to the telemetry metrics dictionary and the CSV logger.

**3. Update `run.py`:**
*   Load the BAO data via `load_bao_data()`.
*   Pass it into `train_model()`.

## 4. Testing, Validation & CI/CD

Before finalizing the feature, the following engineering standards must be met:

### 4.1. Unit Testing
*   **Create `tests/unit/test_bao.py`**
*   Initialize a mock `MetricNN` frozen to a standard isotropic FLRW state.
*   Verify that `get_bao_distances` returns perfectly isotropic results (e.g., $D_M$ along the x-axis matches $D_M$ along the z-axis).
*   Verify that the BAO loss function executes without JAX shape errors or NaNs.

### 4.2. Integration Testing
*   **Update `tests/integration/test_control_training.py`**
*   Modify the control tests to pass a mock BAO dataset into the `train_model` function to ensure the addition of the BAO loss does not break the global training loop or the optimizer state.

### 4.3. Code Quality (Pre-Commit)
*   The entire codebase must adhere to the style guidelines defined in `technical_design.md`.
*   Run `pre-commit run --all-files` locally to ensure:
    *   `ruff` passes without linting errors.
    *   Code is formatted correctly (2-space indent, max 80 chars).
    *   JAX type hints (`jnp.ndarray`) are correctly applied to the new BAO functions.
