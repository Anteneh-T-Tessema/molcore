#!/usr/bin/env bash
# Sets up the molcore development environment.
# Run once: ./setup_dev.sh
# Activate afterwards: source .venv/bin/activate

set -euo pipefail

# Resolve Python — prefer 3.11, fall back to 3.12, then whatever python3 is.
# Homebrew keg-only pythons aren't in PATH, so check Homebrew prefix directly.
HOMEBREW_PREFIX="${HOMEBREW_PREFIX:-/opt/homebrew}"

if [ -z "${PYTHON:-}" ]; then
    if command -v python3.12 &>/dev/null; then
        PYTHON=python3.12
    elif [ -x "$HOMEBREW_PREFIX/opt/python@3.12/bin/python3.12" ]; then
        PYTHON="$HOMEBREW_PREFIX/opt/python@3.12/bin/python3.12"
    else
        PYTHON=python3
    fi
fi

PYVER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "==> Using $PYTHON ($PYVER)"

if [[ "$PYVER" < "3.11" ]]; then
    echo "ERROR: Python 3.11+ required (got $PYVER). torch wheels don't ship for older versions."
    exit 1
fi

echo "==> Creating .venv with $PYTHON"
"$PYTHON" -m venv .venv
source .venv/bin/activate

echo "==> Upgrading pip + build tools"
pip install --upgrade pip wheel

echo "==> Installing maturin (Rust→Python build backend)"
pip install "maturin>=1.7,<2.0"

echo "==> Installing torch + numpy (CPU-only, fastest install)"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy

echo "==> Installing RDKit"
pip install rdkit

echo "==> Installing PyG (torch-geometric)"
pip install torch-geometric

echo "==> Installing dev tools"
pip install pytest pytest-benchmark

echo "==> Installing optional pretrained wrappers (skippable)"
pip install molfeat datamol || echo "  (molfeat/datamol skipped — optional)"

echo "==> Installing Rust stable (if not present)"
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"

echo "==> Building molcore Rust extension in dev mode"
maturin develop --release --features extension-module

echo "==> Running Rust unit tests"
cargo test -p molcore-core -q

echo "==> Running Python + eval tests"
pytest tests/python evals/ -q

echo ""
echo "All done. Activate with: source .venv/bin/activate"
echo "Commands:"
echo "  cargo test -p molcore-core        # Rust unit tests"
echo "  pytest tests/python evals/ -v    # Python integration + skill evals"
echo "  python benchmarks/bench_fingerprints.py   # throughput benchmark"
