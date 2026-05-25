@../GEMINI.md

# Engineering Standards for Lumpyspace

## Python Development
- **Dependencies:** Never use `pip install <package>` directly. Always add dependencies to `pyproject.toml` (under `project.dependencies` or `project.optional-dependencies.dev`) and install using `pip install -e .` or `pip install -e ".[dev]"`.
- **Linting & Formatting:** `pre-commit` must pass before any task is considered complete.
- **Testing:** 100% coverage for core math is required. Always use `pytest`.

## Technical Constants
- **JAX:** Use `jnp` (jax.numpy) for mathematical operations.
- **PINN Architecture:** SiREN activations should use `jnp.sin`.
- **Metric Signature:** Lorentzian (-, +, +, +) must be strictly enforced in `MetricNN`.

