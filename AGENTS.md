# Repository Guidelines

## Project Structure & Module Organization

This Python 3.12 package implements Isaac Lab locomotion tasks for a Unitree Go2 carrying randomized payloads.

- `src/go2_payload_transport/tasks/payload_balance/mdp/`: rewards, observations, and curricula. Export new terms through imports and `__all__` in `mdp/__init__.pyi`, which drives lazy loading.
- `tasks/payload_balance/config/go2/` within the package: environment configuration, Gym registrations, and `agents/` training configurations.
- `tests/`: pytest registration checks; `scripts/`: environment discovery and runtime setup.
- `config/extension.toml`: Isaac Sim extension metadata. Robot assets come from Isaac Lab; generated training artifacts belong in ignored `logs/`, `outputs/`, or `videos/`.

## Build, Test, and Development Commands

Use a dedicated shell and select the compatible Isaac Lab source environment before running tasks:

```bash
source scripts/use-isaaclab-source.sh
uv run python scripts/list_envs.py --show_presets
uv run --with pytest pytest
uv run --with pre-commit pre-commit run --all-files
TASK=Go2PayloadTransport-Payload-Balance-GO2
uv run isaaclab train --rl_library rsl_rl --task "$TASK" --max_iterations 10 --num_envs 64
uv run isaaclab play --rl_library rsl_rl --task "$TASK" --checkpoint latest --num_envs 16
```

These commands list environments, run tests and quality checks, smoke-test training, and replay a checkpoint. The setup defaults to `~/IsaacLab/.venv`; override with `ISAACLAB_SOURCE_VENV`. The pinned wheel has incompatible imports and may silently ignore training commands. `uv build` builds the package; commit dependency changes with `uv.lock`.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, `PascalCase` configuration classes, and uppercase constants. Ruff targets Python 3.12 with 120-character lines and project-specific import groups. Pre-commit runs Ruff lint/format, codespell, and file checks. Preserve existing license headers. Inspect upstream configuration classes before adding overrides or new MDP terms.

## Testing Guidelines

Name tests `tests/test_*.py` and functions `test_*`. Available markers are `unit`, `integration`, `smoke`, and `kitless`; no coverage threshold is configured. Registration tests validate entry-point strings only. After configuration edits, also import and instantiate `UnitreeGo2PayloadFlatEnvCfg`; use a short GPU training run for behavior changes. The Circle task is evaluation-only.

## Commit & Pull Request Guidelines

History uses short, descriptive subjects such as “Point uv run at the ~/IsaacLab source checkout”; no enforced prefix convention is evident. Keep commits focused. PRs should explain the change, relevant task/configuration effects, validation commands and results, and linked issues when applicable. Include run metrics or videos for locomotion changes; keep checkpoints and generated logs out of Git.
