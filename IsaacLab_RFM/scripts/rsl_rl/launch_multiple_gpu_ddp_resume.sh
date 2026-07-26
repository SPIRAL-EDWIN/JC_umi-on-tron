#!/usr/bin/env bash

set -euo pipefail

check_only=false
if [[ "${1:-}" == "--check" ]]; then
    check_only=true
    shift
fi
if (( $# != 0 )); then
    echo "Usage: $0 [--check]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_RFM_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export WBC_LOG_ROOT="${WBC_LOG_ROOT:-${ISAACLAB_RFM_DIR}/logs/rsl_rl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-v0}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
RUN_NAME="${RUN_NAME:-no_curr}"
NUM_ENVS_PER_GPU="${NUM_ENVS_PER_GPU:-6144}"
MAX_ITERATIONS="${MAX_ITERATIONS:-15000}"
MASTER_PORT="${MASTER_PORT:-29500}"
LAUNCH_STAMP="${LAUNCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"

IFS=',' read -r -a selected_gpus <<<"${GPU_IDS}"
if (( ${#selected_gpus[@]} < 2 )); then
    echo "ERROR: DDP requires at least two GPU IDs; got GPU_IDS=${GPU_IDS}." >&2
    exit 2
fi
for gpu in "${selected_gpus[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid GPU ID '${gpu}' in GPU_IDS=${GPU_IDS}." >&2
        exit 2
    fi
done
NPROC_PER_NODE="${#selected_gpus[@]}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is not available." >&2
    exit 1
fi
gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l)"
compute_pids=()
for gpu in "${selected_gpus[@]}"; do
    if (( gpu >= gpu_count )); then
        echo "ERROR: GPU ${gpu} does not exist; detected GPU count is ${gpu_count}." >&2
        exit 1
    fi
    while IFS= read -r pid; do
        [[ -n "${pid}" ]] && compute_pids+=("${pid}")
    done < <(
        nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
            | sed -n 's/^[[:space:]]*\([0-9][0-9]*\).*$/\1/p'
    )
done
if (( ${#compute_pids[@]} > 0 )); then
    echo "ERROR: selected GPUs ${GPU_IDS} already have compute processes; refusing to oversubscribe them." >&2
    ps -o pid=,user=,etime=,args= -p "$(IFS=,; echo "${compute_pids[*]}")" >&2 || true
    exit 1
fi

if ! "${PYTHON_BIN}" -c "import isaaclab, torch; assert torch.distributed.is_nccl_available()" >/dev/null 2>&1; then
    echo "ERROR: activate an Isaac Lab environment with PyTorch NCCL support first." >&2
    exit 1
fi

if [[ "${check_only}" == true ]]; then
    echo "Preflight passed: GPUs=${GPU_IDS}, ranks=${NPROC_PER_NODE}"
    echo "Mode: fresh training"
    exit 0
fi

cd "${SCRIPT_DIR}"
echo "Launching synchronous DDP on physical GPUs ${GPU_IDS} (${NPROC_PER_NODE} ranks)."
echo "Mode: fresh training"
# echo "Safety weight: ${SAFETY_WEIGHT}"
exec "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --master-port="${MASTER_PORT}" \
    ios_train.py \
    --distributed \
    --headless \
    --task="${TASK}" \
    --run_name="${RUN_NAME}_${LAUNCH_STAMP}" \
    --seed=42 \
    --asset_usd_dir="/tmp/IsaacLab/ddp${NPROC_PER_NODE}_${LAUNCH_STAMP}" \
    --num_envs="${NUM_ENVS_PER_GPU}" \
    --max_iterations="${MAX_ITERATIONS}" \
    --logger=wandb \
    "agent.wandb_run_name=${RUN_NAME}_${LAUNCH_STAMP}"
