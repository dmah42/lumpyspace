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

## GPU Contention & Integration/E2E Tests

- **NEVER** execute integration (`tests/integration/`) or end-to-end (`tests/e2e/`) tests without first checking if the GPU is busy.
