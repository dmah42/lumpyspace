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
- **Commit Process:**
  - ALWAYS run relevant tests first.
  - If tests pass, stage files and run `pre-commit` (do NOT use `--all-files` unless requested).
  - Only if `pre-commit` passes, proceed to commit.
  - NEVER combine tests, pre-commit, and git commands into a single shell command string or tool call.

## Technical Constants
- **JAX:** Use `jnp` (jax.numpy) for mathematical operations.
- **PINN Architecture:** SiREN activations should use `jnp.sin`.
- **Metric Signature:** Lorentzian (-, +, +, +) must be strictly enforced in `MetricNN`.

## Documentation & Text
- **Formatting & Style:** Do NOT use em-dashes (—) in any documentation or text. Ever.

## Debugging
- Never guess. Add debug logs, check the full output of tests, evaluate the
  problem and potential solutions before trying to fix something.
