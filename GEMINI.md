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

## State & Checkpointing
- **Resumption Requirement:** Any time a stateful parameter is added to the training loop (e.g., AdaptivePenaltyState, EMA variables), it MUST be explicitly saved to the checkpoint `.meta` file and flawlessly restored upon `resume=True`.

## Pre-commit Workflow
- Always stage intended changes (`git add <files>`) before running `pre-commit`.
- Run pre-commit simply via `pre-commit run` so it operates strictly on staged changes.
- **NEVER** run pre-commit with `--all-files`.
- **NEVER** pass specific files via `--files ...`.

## Read Code Before Modifying (Zero Assumptions)
- **NEVER** modify, replace, or overwrite code in any file without reading the latest content of the target lines with `view_file` FIRST.
- Never assume, predict, or guess what the user or other processes have written in the file. Always inspect the active disk state before proposing any edit.

## GPU Contention & Integration/E2E Tests
- **NEVER** execute integration (`tests/2.integration/`) or end-to-end (`tests/3.e2e/`) tests without first checking if the GPU is busy.

## Choice of words
- **NEVER** blacklist, always denylist
- **NEVER** whitelist, always allowlist

## Commit Messages
- **NEVER** use commit prefixes (e.g., `feat:`, `feat(...)`, `fix:`, `chore:`, `refactor:`, etc.).
- Write commit titles in plain, direct English (imperative mood or concise summary).

## Answering Questions (Zero Extraneous Tool Calls)
- When asked a conceptual, architectural, or clarifying question, answer directly from the active context and code base logic.
- **NEVER** write or execute scratch scripts, database inspection scripts, or exploratory CLI commands simply to answer a user question unless explicitly instructed.
- Minimize token burn and tool churn: do not gather runtime telemetry or live state when answering questions about code behavior or control flow.