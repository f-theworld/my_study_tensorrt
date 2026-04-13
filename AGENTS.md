# AGENTS.md

## Cursor Cloud specific instructions

### Repository overview

`my_study_tensorrt` is a Python-based study/learning project for NVIDIA TensorRT. The repository is in early stages and does not yet contain application code or services.

### Development environment

- **Python**: 3.12 (system Python on Ubuntu 24.04)
- **Linter**: `ruff` — run `ruff check .` from the workspace root
- **Test framework**: `pytest` — run `pytest` from the workspace root
- **No GPU/CUDA**: The Cloud Agent VM does not have an NVIDIA GPU or CUDA toolkit installed. TensorRT inference cannot be tested on the VM directly. Focus on CPU-compatible code and unit tests.

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
