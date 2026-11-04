#!/usr/bin/env bash
set -euo pipefail

# Repository dir. Resolved to an absolute path: the script cds into it partway
# through, so a relative argument would make every path built from it — the
# output file included — resolve against the wrong directory from then on.
REPO_DIR="$(cd "${1:-$(pwd)}" && pwd)"

# Prefer using the env's python/pip directly (avoid relying on activation)
PY_BIN="$REPO_DIR/.venv/bin/python"
PIP_BIN="$REPO_DIR/.venv/bin/pip"
ACTIVATE="$REPO_DIR/.venv/bin/activate"

if [ ! -x "$PY_BIN" ] || [ ! -x "$PIP_BIN" ]; then
  # Try sourcing activate as a fallback (keeps original behaviour)
  if [ -f "$ACTIVATE" ]; then
    # shellcheck source=/dev/null
    echo "source $ACTIVATE"
    source "$ACTIVATE" || {
      echo "Error: Could not activate virtual environment."
      exit 1
    }
    PY_BIN="$(which python)"
  else
    echo "Error: Python/pip not found in virtualenv and activate script missing."
    exit 1
  fi
fi

# Pick the torch build. pyproject.toml locks a CUDA and a CPU resolution as two
# conflicting dependency groups; a lockfile cannot know about the hardware, so
# the choice is made here. Fall back to the CPU wheels when there is no usable
# GPU (login node, CI runner, laptop) — nvidia-smi must both exist and report a
# device, since the binary is present but fails on Snellius login nodes.
SYNC_ARGS=()
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
  echo "GPU detected, syncing the CUDA build"
else
  echo "No GPU detected, falling back to the CPU build"
  SYNC_ARGS=(--no-group cuda --group cpu)
fi

# Run sync env from $REPO_DIR
echo "cd $REPO_DIR && uv sync ${SYNC_ARGS[*]}"
cd "$REPO_DIR" || exit # '|| exit' ensures the script stops if cd fails
uv sync "${SYNC_ARGS[@]}"

# Determine Python major.minor version from the env
echo ""$PY_BIN" -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))'"
PY_VER=$("$PY_BIN" -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))')

# Determine the torch version/CUDA build actually resolved by uv, so the
# conda env mirrors it instead of installing a different, stale pin.
TORCH_VERSION_FULL=$("$PY_BIN" -c 'import torch; print(torch.__version__)')
TORCH_VERSION="${TORCH_VERSION_FULL%%+*}"
TORCH_CUDA_TAG="${TORCH_VERSION_FULL#*+}"
if [ "$TORCH_CUDA_TAG" = "$TORCH_VERSION_FULL" ]; then
  TORCH_CUDA_TAG="cpu"
fi
echo "Detected torch=$TORCH_VERSION (index tag: $TORCH_CUDA_TAG)"

# torchvision is optional: assignment 0 only needs torch. Leave it out when uv did
# not resolve it; when it is installed, a failing import still aborts the script.
TORCHVISION_VERSION=""
if "$PY_BIN" -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("torchvision") is None)'; then
  TORCHVISION_VERSION_FULL=$("$PY_BIN" -c 'import torchvision; print(torchvision.__version__)')
  TORCHVISION_VERSION="${TORCHVISION_VERSION_FULL%%+*}"
  echo "Detected ($TORCHVISION_VERSION_FULL) torchvision=$TORCHVISION_VERSION"
else
  echo "No torchvision in the resolved environment, leaving it out"
fi

# Get pip freeze output (keeps VCS and editable lines)
echo "uv pip list --format freeze"
PIP_FREEZE=$(uv pip list --format freeze)

# The five packages that have to come from the PyG find-links page rather than
# from an index. Only assignment 2 uses them; the others resolve without any of
# them, and emitting the block there put five unused packages, and a find-links
# page to fetch them from, into environments that never import PyG.
PYG_PACKAGES=(pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv)
if echo "$PIP_FREEZE" | grep -qE '^(torch-geometric|pyg-lib)=='; then
  USES_PYG=1
  echo "PyG detected, including the find-links block"
else
  USES_PYG=0
  echo "No PyG in the resolved environment, skipping the find-links block"
fi

# Remove problematic packages — torch will pull in the correct
# nvidia-*/cuda versions itself, pinning them causes solver conflicts.
PIP_FREEZE=$(echo "$PIP_FREEZE" | grep -vE '^(nvidia-|cuda-toolkit==|cuda-bindings==|cuda-pathfinder==|triton==)')

# Drop everything the block below names explicitly. Without this each of them is
# requested twice in one requirements list, once unpinned from the block and
# once pinned to a local version by the freeze (torchvision==0.24.0 next to
# torchvision==0.24.0+cu128), which is two conflicting requirements for the same
# package. torch was already handled this way; the rest were not.
EXPLICIT=(torch torchvision)
if [ "$USES_PYG" -eq 1 ]; then
  EXPLICIT+=("${PYG_PACKAGES[@]}")
fi
for pkg in "${EXPLICIT[@]}"; do
  PIP_FREEZE=$(echo "$PIP_FREEZE" | grep -v "^${pkg}==")
done

CONDA_YAML_PATH="$REPO_DIR/conda_environment.yaml"

# Write conda YAML
{
  echo "name: uvadlc_seminars"
  echo "channels:"
  echo "  - conda-forge"
  echo "  - defaults"
  echo "dependencies:"
  echo "  - python=$PY_VER"
  echo "  - pip"
  echo "  - pip:"
  # ---- PyTorch (matches the build resolved by uv, in this case 2.9.0+cu128) ----
  echo "    - \"--index-url=https://download.pytorch.org/whl/$TORCH_CUDA_TAG\""
  echo '    - "--extra-index-url=https://pypi.org/simple"'
  echo "    - \"torch==$TORCH_VERSION\""
  if [ -n "$TORCHVISION_VERSION" ]; then
    echo "    - \"torchvision==$TORCHVISION_VERSION\""
  fi
  # ---- PyG ecosystem (same find-links page uv actually resolved against) ----
  # Only for projects that actually resolved PyG; see USES_PYG above.
  # Remove torch-spline-conv from PYG_PACKAGES if using torch>2.9.
  if [ "$USES_PYG" -eq 1 ]; then
    # Built from the tag rather than TORCH_VERSION_FULL: macOS torch has no
    # local version (plain 2.9.0), but its wheels are on the +cpu page.
    echo "    - \"--find-links=https://data.pyg.org/whl/torch-$TORCH_VERSION+$TORCH_CUDA_TAG.html\""
    for pkg in "${PYG_PACKAGES[@]}"; do
      printf '    - "%s"\n' "$pkg"
    done
  fi

  if [ -z "$PIP_FREEZE" ]; then
    echo "    # no additional pip packages detected"
  else
    while IFS= read -r pkg; do
      [ -z "$pkg" ] && continue
      printf '    - "%s"\n' "$(printf '%s' "$pkg" | sed 's/"/\\"/g')"
    done <<< "$PIP_FREEZE"
  fi
} > "$CONDA_YAML_PATH"

echo "Generated conda environment YAML at: $CONDA_YAML_PATH"

# Deactivate current uv env
echo "deactivate"
deactivate || true