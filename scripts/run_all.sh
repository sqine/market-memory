#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q
python scripts/build_notebooks.py
python - <<'PY'
import nbformat
from nbclient import NotebookClient
for p in ["notebooks/01_hurst.ipynb", "notebooks/02_chaos.ipynb"]:
    nb = nbformat.read(p, as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": "notebooks"}}).execute()
    nbformat.write(nb, p)
PY
