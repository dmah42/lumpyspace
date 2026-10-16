@../AGENTS.md

# Engineering Standards for Lumpyspace

## Testing

- **Coverage:** 100% coverage for core math is required.

## Technical Constants

- **JAX:** Use `jnp` (jax.numpy) for mathematical operations.
- **PINN Architecture:** SiREN activations should use `jnp.sin`.
- **Metric Signature:** Lorentzian (-, +, +, +) must be strictly enforced in `MetricNN`.

## State & Checkpointing

- **Resumption Requirement:** Any time a stateful parameter is added to the training loop (e.g., AdaptivePenaltyState, EMA variables), it MUST be explicitly saved to the checkpoint `.meta` file and flawlessly restored upon `resume=True`.

## Training Metrics Log

- `logs/training_metrics.csv` holds per-step training metrics only: the loss and its terms, `omega_m`, and any score the run is judged by.
- Diagnostics and telemetry (penalty bump/decay events, `w_penalty` values, reachability fractions) go to a separate channel such as stdout, a sidecar log like `logs/penalty_events.csv`, or the checkpoint `.meta`.

## GPU Contention & Integration/E2E Tests

- **NEVER** execute integration (`tests/integration/`) or end-to-end (`tests/e2e/`) tests without first checking if the GPU is busy.
