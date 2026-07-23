#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_RFM_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"

export WBC_LOG_ROOT="${WBC_LOG_ROOT:-/media/edwin/ChenJing26/WBC_logs}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON_BIN="$(command -v "${PYTHON_BIN:-python}")"
STARTUP_GRACE_SECONDS="${STARTUP_GRACE_SECONDS:-60}"
task="Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-v0"
resume_run="2026-07-19_23-46-01_reward_set_gpu1_20260719_234554"
resume_checkpoint="model_9000.pt"

foreground=false
if [[ "${1:-}" == "--foreground" ]]; then
    foreground=true
    shift
fi
if (( $# != 0 )); then
    echo "Usage: $0 [--foreground]" >&2
    exit 2
fi

if [[ ! "${STARTUP_GRACE_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: STARTUP_GRACE_SECONDS must be a positive integer." >&2
    exit 2
fi

preflight() {
    local gpu_count
    local -a training_pids=()
    local -a compute_pids=()

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi is not available." >&2
        return 1
    fi
    gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l)"
    if (( gpu_count < 4 )); then
        echo "ERROR: Four GPUs are required, but only ${gpu_count} were detected." >&2
        return 1
    fi

    mapfile -t training_pids < <(pgrep -u "$(id -u)" -f '[i]os_train\.py' 2>/dev/null || true)
    if (( ${#training_pids[@]} > 0 )); then
        echo "ERROR: Training is already running; refusing a duplicate four-GPU launch." >&2
        ps -o pid=,etime=,args= -p "$(IFS=,; echo "${training_pids[*]}")" >&2 || true
        return 1
    fi

    mapfile -t compute_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed -n 's/^[[:space:]]*\([0-9][0-9]*\).*$/\1/p' | sort -u)
    if (( ${#compute_pids[@]} > 0 )); then
        echo "ERROR: GPUs already have compute processes; refusing to oversubscribe them." >&2
        ps -o pid=,user=,etime=,args= -p "$(IFS=,; echo "${compute_pids[*]}")" >&2 || true
        return 1
    fi
}

if ! "${PYTHON_BIN}" -c "import isaaclab" >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON_BIN} cannot import isaaclab." >&2
    echo "Activate the training environment first:" >&2
    echo "  conda activate isaaclab_umi_on_tron" >&2
    exit 1
fi

launch_stamp="${LAUNCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
session_name="reward_sweep_${launch_stamp}"

# Outside tmux, relaunch this script in a detached session. The absolute Python
# path preserves the currently activated Isaac Lab environment.
if [[ "${foreground}" == false && -z "${TMUX:-}" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "ERROR: tmux is not installed. Run with --foreground or install tmux." >&2
        exit 1
    fi

    preflight

    printf -v tmux_command \
        'exec env PYTHON_BIN=%q WBC_LOG_ROOT=%q PYTORCH_CUDA_ALLOC_CONF=%q STARTUP_GRACE_SECONDS=%q LAUNCH_STAMP=%q bash %q --foreground' \
        "${PYTHON_BIN}" \
        "${WBC_LOG_ROOT}" \
        "${PYTORCH_CUDA_ALLOC_CONF}" \
        "${STARTUP_GRACE_SECONDS}" \
        "${launch_stamp}" \
        "${SCRIPT_PATH}"

    tmux new-session -d -s "${session_name}" -c "${SCRIPT_DIR}" "${tmux_command}"
    echo "Preflight passed. Verifying all four workers for ${STARTUP_GRACE_SECONDS}s; do not launch again."
    sleep "$((STARTUP_GRACE_SECONDS + 3))"
    if ! tmux has-session -t "${session_name}" 2>/dev/null; then
        echo "ERROR: Training session exited during startup." >&2
        echo "No launcher directory was kept for this failed attempt." >&2
        exit 1
    fi
    echo "Four-GPU reward sweep started in detached tmux session: ${session_name}"
    echo "Attach:  tmux attach -t ${session_name}"
    echo "Detach:  Ctrl-b, then d"
    echo "Status:  tmux has-session -t ${session_name} && echo running"
    echo "Stop:    tmux kill-session -t ${session_name}"
    exit 0
fi

if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock is required to prevent duplicate launches." >&2
    exit 1
fi
lock_file="${TMPDIR:-/tmp}/launch_4gpu_reward_sweep.lock"
exec 9>"${lock_file}"
if ! flock -n 9; then
    echo "ERROR: Another four-GPU launcher is starting or still running." >&2
    exit 1
fi
preflight

# One reward configuration per physical GPU. Edit these rows before launching.
#                        GPU 0  GPU 1  GPU 2  GPU 3
position_weights=(        3.0    4.0    3.0    3.0 )
orientation_weights=(     3.0    5.0    3.0    3.0 )
pb_weights=(             15.0   15.0   15.0   20.0 )
reference_weights=(       5.0    5.0    5.0    5.0 )
action_weights=(         -1.0   -1.0   -1.0   -1.0 )
foot_slip_weights=(      -2.0   -2.0   -2.0   -2.0 )
foot_contacts_reg=(       0.5    0.5    0.5    0.5 )
safety_weights=(          3.0    3.0    2.0    3.0 )

num_envs=8192
max_iterations=10000
console_log_dir="${WBC_LOG_ROOT}/launcher_${launch_stamp}"
pids=()

resume_path="${WBC_LOG_ROOT}/${resume_run}/${resume_checkpoint}"
if [[ ! -f "${resume_path}" ]]; then
    echo "ERROR: Resume checkpoint does not exist: ${resume_path}" >&2
    exit 1
fi

if [[ -e "${console_log_dir}" ]] || find "${WBC_LOG_ROOT}" -mindepth 1 -maxdepth 1 -name "*_${launch_stamp}" -print -quit 2>/dev/null | grep -q .; then
    echo "ERROR: Launch stamp ${launch_stamp} already has logs; choose a new stamp." >&2
    exit 1
fi

launcher_valid=false
cleanup_failed_startup() {
    local status=$?
    if [[ "${launcher_valid}" == false ]]; then
        for pid in "${pids[@]}"; do
            kill "${pid}" 2>/dev/null || true
        done
        for pid in "${pids[@]}"; do
            wait "${pid}" 2>/dev/null || true
        done
        rm -rf -- "${console_log_dir}"
        find "${WBC_LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "*_${launch_stamp}" -exec rm -rf -- {} + 2>/dev/null || true
        find /tmp/IsaacLab -mindepth 1 -maxdepth 1 -type d -name "${launch_stamp}_gpu*" -exec rm -rf -- {} + 2>/dev/null || true
    fi
    return "${status}"
}
trap cleanup_failed_startup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "${console_log_dir}"

cd "${SCRIPT_DIR}"

# Which GPU to use for each run. The CUDA_VISIBLE_DEVICES environment variable is set
for gpu in 0 1 2 3; do
    run_name="reward_set_gpu${gpu}_${launch_stamp}"
    console_log="${console_log_dir}/${run_name}.log"
    usd_dir="/tmp/IsaacLab/${launch_stamp}_gpu${gpu}"

    echo "Launching ${run_name} on physical GPU ${gpu}"
    echo "  position=${position_weights[$gpu]}, orientation=${orientation_weights[$gpu]}, pb=${pb_weights[$gpu]}, reference=${reference_weights[$gpu]}"
    echo "  action_rate=${action_weights[$gpu]}, foot_slip=${foot_slip_weights[$gpu]}, foot_contacts=${foot_contacts_reg[$gpu]}, safety=${safety_weights[$gpu]}"
    echo "  resume=${resume_path}"
    echo "  usd_dir=${usd_dir}"
    echo "  console=${console_log}"

    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        exec "${PYTHON_BIN}" ios_train.py \
            --headless \
            --device=cuda:0 \
            --task="${task}" \
            --run_name="${run_name}" \
            --seed=42 \
            --asset_usd_dir="${usd_dir}" \
            --num_envs="${num_envs}" \
            --max_iterations="${max_iterations}" \
            --resume=True \
            --load_run="${resume_run}" \
            --checkpoint="${resume_checkpoint}" \
            --logger=wandb \
            "env.rewards.track_EE_position_exp.weight=${position_weights[$gpu]}" \
            "env.rewards.track_EE_orientation_exp.weight=${orientation_weights[$gpu]}" \
            "env.rewards.track_EE_pb.weight=${pb_weights[$gpu]}" \
            "env.rewards.track_EE_reference_exp.weight=${reference_weights[$gpu]}" \
            "env.rewards.action_rate_l2.weight=${action_weights[$gpu]}" \
            "env.rewards.foot_slip_l2.weight=${foot_slip_weights[$gpu]}" \
            "env.rewards.feet_contacts_reg.weight=${foot_contacts_reg[$gpu]}" \
            "env.rewards.safety_exp.weight=${safety_weights[$gpu]}" \
            "agent.wandb_run_name=${run_name}"
    ) >"${console_log}" 2>&1 &

    pids+=("$!")
    echo "  pid=${pids[-1]}"
done

echo "Waiting ${STARTUP_GRACE_SECONDS}s to confirm that all four workers survive startup."
deadline=$((SECONDS + STARTUP_GRACE_SECONDS))
while (( SECONDS < deadline )); do
    for index in "${!pids[@]}"; do
        pid="${pids[$index]}"
        state="$(ps -o stat= -p "${pid}" 2>/dev/null | cut -c1 || true)"
        if [[ -z "${state}" || "${state}" == "Z" ]]; then
            echo "ERROR: GPU ${index} worker (pid=${pid}) exited during startup." >&2
            exit 1
        fi
    done
    sleep 1
done

for gpu in 0 1 2 3; do
    if ! find "${WBC_LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "*_reward_set_gpu${gpu}_${launch_stamp}" -print -quit 2>/dev/null | grep -q .; then
        echo "ERROR: GPU ${gpu} did not create its training run directory during startup." >&2
        exit 1
    fi
done

launcher_valid=true
echo "All four jobs passed startup validation and are running."
echo "Follow one job with: tail -f ${console_log_dir}/reward_set_gpu0_${launch_stamp}.log"

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

exit "${status}"
