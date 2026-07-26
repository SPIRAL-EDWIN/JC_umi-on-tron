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
RESUME_RUN="${RESUME_RUN:-2026-07-23_22-53-58_reward_set_gpu0_20260723_225351}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-model_18000.pt}"
# Prefer the single-GPU spelling. Also accept GPU_IDS for consistency with the
# DDP launcher, as long as it contains exactly one physical GPU ID.
GPU_ID="${GPU_ID:-${GPU_IDS:-2}}"
SAFETY_WEIGHT="${SAFETY_WEIGHT:-2.0}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
LAUNCH_STAMP="${LAUNCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
safety_tag="${SAFETY_WEIGHT//./p}"
RUN_NAME="${RUN_NAME:-gpu${GPU_ID}_safety${safety_tag}_resume_gpu0_18000}"

if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GPU_ID (or GPU_IDS) must contain exactly one non-negative GPU ID." >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is not available." >&2
    exit 1
fi
gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l)"
if (( GPU_ID >= gpu_count )); then
    echo "ERROR: GPU ${GPU_ID} does not exist; detected GPU count is ${gpu_count}." >&2
    exit 1
fi
mapfile -t compute_pids < <(
    nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | sed -n 's/^[[:space:]]*\([0-9][0-9]*\).*$/\1/p'
)
if (( ${#compute_pids[@]} > 0 )); then
    echo "ERROR: GPU ${GPU_ID} already has a compute process." >&2
    ps -o pid=,user=,etime=,args= -p "$(IFS=,; echo "${compute_pids[*]}")" >&2 || true
    exit 1
fi

resume_path="${WBC_LOG_ROOT}/${RESUME_RUN}/${RESUME_CHECKPOINT}"
if [[ ! -f "${resume_path}" ]]; then
    echo "ERROR: checkpoint does not exist: ${resume_path}" >&2
    exit 1
fi
if ! "${PYTHON_BIN}" -c "import isaaclab" >/dev/null 2>&1; then
    echo "ERROR: activate the Isaac Lab training environment first." >&2
    exit 1
fi

if [[ "${check_only}" == true ]]; then
    echo "Preflight passed: GPU=${GPU_ID}, safety_weight=${SAFETY_WEIGHT}"
    echo "Resume: ${resume_path}"
    exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" ios_train.py \
    --headless \
    --device=cuda:0 \
    --task="${TASK}" \
    --run_name="${RUN_NAME}_${LAUNCH_STAMP}" \
    --seed=42 \
    --asset_usd_dir="/tmp/IsaacLab/${RUN_NAME}_${LAUNCH_STAMP}" \
    --num_envs="${NUM_ENVS}" \
    --max_iterations="${MAX_ITERATIONS}" \
    --resume=True \
    --load_run="${RESUME_RUN}" \
    --checkpoint="${RESUME_CHECKPOINT}" \
    --logger=wandb \
    env.rewards.track_EE_position_exp.weight=3.0 \
    env.rewards.track_EE_orientation_exp.weight=3.0 \
    env.rewards.track_EE_pb.weight=15.0 \
    env.rewards.track_EE_reference_exp.weight=5.0 \
    "env.rewards.safety_exp.weight=${SAFETY_WEIGHT}" \
    "agent.wandb_run_name=${RUN_NAME}_${LAUNCH_STAMP}"
