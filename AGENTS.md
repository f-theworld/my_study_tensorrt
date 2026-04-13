# AGENTS.md

## Cursor Cloud specific instructions

### Repository overview

`my_study_tensorrt` is a Python-based study/learning project for NVIDIA TensorRT. The repository is in early stages and does not yet contain application code or services.

### Development environment

- **Python**: 3.12 (system Python on Ubuntu 24.04)
- **trtpy**: 1.2.6 — TensorRT Python helper (`python3 -m trtpy` for CLI, `import trtpy` in code)
- **Linter**: `ruff` — run `ruff check .` from the workspace root
- **Test framework**: `pytest` — run `pytest` from the workspace root
- **No GPU/CUDA**: The Cloud Agent VM does not have an NVIDIA GPU or CUDA toolkit installed. TensorRT inference cannot be tested on the VM directly. `trtpy` runs in `NoInit` (sysmode) — module imports work but GPU-dependent features will fail.

### trtpy notes

- CLI entry point: `python3 -m trtpy <subcommand>` (the `trtpy` shell command may not be on PATH; always use the module form)
- Environment was initialized with `python3 -m trtpy get-env --cuda=11`
- Import convention: `import trtpy.init_default as trtpy`

### Running commands

- Ensure `$HOME/.local/bin` is on `PATH` (pip `--user` installs go there):
  ```
  export PATH="$HOME/.local/bin:$PATH"
  ```
- Lint: `ruff check .`
- Test: `pytest`

### Notes

- There are no services, Docker containers, or databases to start.
- When Python dependencies are added (e.g., `requirements.txt` or `pyproject.toml`), the update script will install them automatically on VM startup.
