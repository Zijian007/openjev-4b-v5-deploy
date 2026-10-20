#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
backend=${1:-}
accelerate=${2:-}
if [[ "$backend" != custom && "$backend" != sglang ]] || \
   [[ -n "$accelerate" && "$accelerate" != --accelerate ]] || \
   [[ "$backend" == sglang && -n "$accelerate" ]] || [[ $# -gt 2 ]]; then
    echo 'Usage: bash scripts/setup.sh custom [--accelerate] | sglang' >&2
    exit 2
fi
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || {
    echo 'The GPU server requires Linux x86_64.' >&2; exit 1;
}
command -v uv >/dev/null || { echo 'Install uv first (see README).' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'Install an NVIDIA driver first.' >&2; exit 1; }
target=.venv
[[ "$backend" == sglang ]] && target=.venv-sglang
if [[ ! -x "$target/bin/python" ]]; then
    uv venv --python 3.12 "$target"
fi
uv pip install --python "$target/bin/python" -r "requirements/$backend.lock.txt"
if [[ "$accelerate" == --accelerate ]]; then
    command -v nvcc >/dev/null || {
        echo 'Optional acceleration needs CUDA Toolkit (nvcc) compatible with Torch CUDA 12.8.' >&2
        echo 'The base environment is installed; omit --accelerate to use it.' >&2
        exit 1
    }
    MAX_JOBS=${MAX_JOBS:-4} uv pip install --python "$target/bin/python" \
        --no-build-isolation --no-deps 'causal-conv1d==1.7.0'
fi
mkdir -p runtime logs
"$target/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable; check driver"; print("Torch:", torch.__version__, "CUDA:", torch.version.cuda); print("GPU:", torch.cuda.get_device_name(0)); x=torch.ones(8,8,device="cuda"); assert (x@x)[0,0].item()==8; print("GPU check passed")'
echo "Installed $backend environment. Next: $target/bin/python download_model.py"
