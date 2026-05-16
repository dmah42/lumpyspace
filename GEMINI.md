# Engineering Standards for Lumpyspace

## Python Development
- **Indentation:** Always use exactly 2 spaces.
- **Dependencies:** Never use `pip install <package>` directly. Always add dependencies to `pyproject.toml` (under `project.dependencies` or `project.optional-dependencies.dev`) and install using `pip install -e .` or `pip install -e ".[dev]"`.
- **Linting & Formatting:** `pre-commit` must pass before any task is considered complete.
- **Testing:** 100% coverage for core math is required. Always use `pytest`.

## Git Workflow
- **Commit Messages:** 
  - Do NOT use prefixes like "feat:", "fix:", or "build:".
  - Keep messages concise and focused on the "why".
  - Do NOT include trivial implementation details (e.g., "fixed indentation", "added 2 spaces").
- **Staging:** Stage only files relevant to the task.

## Technical Constants
- **JAX:** Use `jnp` (jax.numpy) for mathematical operations.
- **PINN Architecture:** SiREN activations should use `jnp.sin`.
- **Metric Signature:** Lorentzian (-, +, +, +) must be strictly enforced in `MetricNN`.
