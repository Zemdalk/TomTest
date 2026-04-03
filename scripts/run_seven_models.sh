#!/usr/bin/env bash
# 固定只跑 BigToM Standard，并将本地模型分配到 GPU 4/5/6/7 并行评测。
# DeepSeek API 模型不占 GPU，也会并行启动。
#
# 用法：
#   bash scripts/run_seven_models.sh
#
# 可选环境变量：
#   TOMTEST=/home/xujy/TomTest
#   DATASET_ROOT=/home/xujy/TomTest/TomDatasets
#   EXTRA_RUN_ARGS='--eval-phase screen --dataset-filter Tomato ToMBench'
#   VLLM_EXTRA='--gpu-memory-utilization 0.85 --max-model-len 8192'

set -euo pipefail

TOMTEST="${TOMTEST:-/home/xujy/TomTest}"
DATASET_ROOT="${DATASET_ROOT:-$TOMTEST/TomDatasets}"
PROMPT_DIR="${PROMPT_DIR:-$TOMTEST/prompt}"
PROMPT_NAME="BigToM Standard"
VLLM_EXTRA="${VLLM_EXTRA:-}"
EXTRA_RUN_ARGS="${EXTRA_RUN_ARGS:-}"

# GPU/端口槽位：4 张卡并行
GPU_IDS=(4 5 6 7)
PORTS=(8004 8005 8006 8007)

# 与 ToM-baseline/run.py 中默认路径一致（本地权重）
QWEN06="/DATA/zhanghy/xai/alignment/SELOR/learning-from-rationales/PRETRAINEDMODEL/Qwen3-0.6B"
QWEN4B="/data/yugx/LongBench/simple_tune/Qwen3-4B"
QWEN8B="/data/yugx/LongBench/simple_tune/Qwen3-8B"
GEMMA="/DATA/xujy/models/gemma-3-4b-it"
LLAMA="/DATA/zhanghy/xai/alignment/SELOR/learning-from-rationales/PRETRAINEDMODEL/Llama-3.1-8B-Instruct"

DEEPSEEK_API="${DEEPSEEK_API:-https://api.deepseek.com/v1}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-50b353ba2b0749bfa62d7e001ee9f9e8}"

LOCAL_MODELS=(
  "$QWEN06|Qwen3-0.6B|Qwen3-0.6B"
  "$QWEN4B|Qwen3-4B|Qwen3-4B"
  "$QWEN8B|Qwen3-8B|Qwen3-8B"
  "$GEMMA|gemma-3-4b-it|gemma-3-4b-it"
  "$LLAMA|Meta-Llama-3.1-8B-Instruct|Meta-Llama-3.1-8B-Instruct"
)

API_MODELS=(
  "deepseek-chat"
  "deepseek-reasoner"
)

cd "$TOMTEST"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERR] 数据集目录不存在: $DATASET_ROOT"
  echo "      可设置 DATASET_ROOT=/home/xujy/ToM-baseline/TomDatasets 或先下载 TomDatasets。"
  exit 1
fi

cleanup_all() {
  jobs -pr | xargs -r kill 2>/dev/null || true
}
trap cleanup_all EXIT INT TERM

wait_vllm_port() {
  local port="$1"
  local tag="$2"
  local n=0
  echo "[INFO] [$tag] 等待 vLLM 监听 :${port} …"
  until curl -s -o /dev/null --connect-timeout 2 "http://127.0.0.1:${port}/v1/models" 2>/dev/null; do
    sleep 3
    n=$((n+1))
    if [[ "$n" -gt 120 ]]; then
      echo "[ERR] [$tag] vLLM 启动超时（port=${port}）。"
      return 1
    fi
  done
  echo "[INFO] [$tag] vLLM 已就绪。"
}

run_one_local_worker() {
  local model_path="$1"
  local served_name="$2"
  local tag="$3"
  local gpu="$4"
  local port="$5"

  if [[ ! -d "$model_path" ]]; then
    echo "[WARN] 跳过（路径不存在）: $model_path ($tag)"
    return 0
  fi

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    local result_dir="$TOMTEST/result/${tag}"
    local api_local="http://127.0.0.1:${port}/v1"
    local serve_log="$result_dir/vllm_serve.log"
    local run_log="$result_dir/run.log"
    mkdir -p "$result_dir"

    echo "========== [$tag] GPU=$gpu PORT=$port =========="
    # shellcheck disable=SC2086
    vllm serve "$model_path" \
      --port "$port" \
      --served-model-name "$served_name" \
      $VLLM_EXTRA \
      >"$serve_log" 2>&1 &
    local vllm_pid=$!

    stop_local() {
      kill "$vllm_pid" 2>/dev/null || true
      wait "$vllm_pid" 2>/dev/null || true
    }
    trap stop_local EXIT

    wait_vllm_port "$port" "$tag"

    # shellcheck disable=SC2086
    python run.py \
      --dataset-root "$DATASET_ROOT" \
      --prompt-dir "$PROMPT_DIR" \
      --result-dir "$result_dir" \
      --prompt-names "$PROMPT_NAME" \
      --model "$served_name" \
      --model-tag "$tag" \
      --api-url "$api_local" \
      --api-key not-needed \
      $EXTRA_RUN_ARGS \
      >"$run_log" 2>&1

    echo "[INFO] [$tag] 完成 -> result/${tag}/"
  ) &

  local pid=$!
  SLOT_PID["$6"]="$pid"
  SLOT_TAG["$6"]="$tag"
}

run_one_api_worker() {
  local name="$1"
  (
    set -euo pipefail
    local result_dir="$TOMTEST/result/${name}"
    local run_log="$result_dir/run.log"
    mkdir -p "$result_dir"
    echo "========== [$name] DeepSeek API =========="
    # shellcheck disable=SC2086
    python run.py \
      --dataset-root "$DATASET_ROOT" \
      --prompt-dir "$PROMPT_DIR" \
      --result-dir "$result_dir" \
      --prompt-names "$PROMPT_NAME" \
      --model "$name" \
      --model-tag "$name" \
      --api-url "$DEEPSEEK_API" \
      --api-key "$DEEPSEEK_API_KEY" \
      $EXTRA_RUN_ARGS \
      >"$run_log" 2>&1
    echo "[INFO] [$name] 完成 -> result/${name}/"
  ) &
  API_PIDS+=("$!")
}

check_finished_slots() {
  local slot pid status
  for slot in "${!GPU_IDS[@]}"; do
    pid="${SLOT_PID[$slot]:-}"
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      set +e
      wait "$pid"
      status=$?
      set -e
      if [[ "$status" -ne 0 ]]; then
        echo "[ERR] [${SLOT_TAG[$slot]}] 任务失败，退出码=$status"
        FAILED=1
      fi
      SLOT_PID["$slot"]=""
      SLOT_TAG["$slot"]=""
      if (( NEXT_LOCAL_IDX < ${#LOCAL_MODELS[@]} )); then
        IFS='|' read -r model_path served_name tag <<< "${LOCAL_MODELS[$NEXT_LOCAL_IDX]}"
        run_one_local_worker "$model_path" "$served_name" "$tag" "${GPU_IDS[$slot]}" "${PORTS[$slot]}" "$slot"
        NEXT_LOCAL_IDX=$((NEXT_LOCAL_IDX + 1))
      fi
    fi
  done
}

declare -a SLOT_PID
declare -a SLOT_TAG
declare -a API_PIDS=()
FAILED=0
NEXT_LOCAL_IDX=0

# 启动两个 API 模型（不占 GPU）
for api_name in "${API_MODELS[@]}"; do
  run_one_api_worker "$api_name"
done

# 首批占满 4 张卡
for slot in "${!GPU_IDS[@]}"; do
  if (( NEXT_LOCAL_IDX >= ${#LOCAL_MODELS[@]} )); then
    break
  fi
  IFS='|' read -r model_path served_name tag <<< "${LOCAL_MODELS[$NEXT_LOCAL_IDX]}"
  run_one_local_worker "$model_path" "$served_name" "$tag" "${GPU_IDS[$slot]}" "${PORTS[$slot]}" "$slot"
  NEXT_LOCAL_IDX=$((NEXT_LOCAL_IDX + 1))
done

# 剩余本地模型：谁先空出 GPU，就接着跑
while (( NEXT_LOCAL_IDX < ${#LOCAL_MODELS[@]} )); do
  set +e
  wait -n
  set -e
  check_finished_slots
done

# 收尾等待所有本地槽位
while :; do
  ACTIVE=0
  for slot in "${!GPU_IDS[@]}"; do
    if [[ -n "${SLOT_PID[$slot]:-}" ]]; then
      ACTIVE=1
      break
    fi
  done
  if [[ "$ACTIVE" -eq 0 ]]; then
    break
  fi
  set +e
  wait -n
  set -e
  check_finished_slots
done

# 等 API 任务
for pid in "${API_PIDS[@]}"; do
  set +e
  wait "$pid"
  status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "[ERR] API 任务失败，退出码=$status"
    FAILED=1
  fi
done

if [[ "$FAILED" -ne 0 ]]; then
  echo "[DONE] 已结束，但至少有一个模型任务失败。请检查各 result/<模型名>/run.log 与 vllm_serve.log。"
  exit 1
fi

echo "[DONE] BigToM Standard 已在 7 个模型上跑完。结果在 $TOMTEST/result/<模型名>/"
