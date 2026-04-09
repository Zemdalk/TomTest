#!/usr/bin/env bash
# 跑 ToMBench：5 个本地模型（不含 deepseek-chat / deepseek-reasoner）。
# 模型列表：
# - Qwen3-0.6B
# - Qwen3-4B
# - Qwen3-8B
# - gemma-3-4b-it
# - Meta-Llama-3.1-8B-Instruct

set -euo pipefail

TOMTEST="${TOMTEST:-/home/xujy/TomTest}"
DATASET_ROOT="${DATASET_ROOT:-$TOMTEST/TomDatasets}"
DATASET_NAME="${DATASET_NAME:-ToMBench}"
GPU_ID="${GPU_ID:-6}"
PORT="${PORT:-8006}"
VLLM_EXTRA="${VLLM_EXTRA:-}"
EXTRA_RUN_ARGS="${EXTRA_RUN_ARGS:-}"

# ToMBench v2 控制项
V2_LANGUAGE="${V2_LANGUAGE:-zh}"
V2_USER_LANGUAGE="${V2_USER_LANGUAGE:-$V2_LANGUAGE}"
V2_COT="${V2_COT:-0}"                     # 1=开启 cot；0=关闭
V2_SHUFFLE_VOTES="${V2_SHUFFLE_VOTES:-5}"
V2_SEED="${V2_SEED:-42}"

QWEN06="${QWEN06:-/DATA/zhanghy/xai/alignment/SELOR/learning-from-rationales/PRETRAINEDMODEL/Qwen3-0.6B}"
QWEN4B="${QWEN4B:-/data/yugx/LongBench/simple_tune/Qwen3-4B}"
QWEN8B="${QWEN8B:-/data/yugx/LongBench/simple_tune/Qwen3-8B}"
GEMMA="${GEMMA:-/DATA/xujy/models/gemma-3-4b-it}"
LLAMA="${LLAMA:-/DATA/zhanghy/xai/alignment/SELOR/learning-from-rationales/PRETRAINEDMODEL/Llama-3.1-8B-Instruct}"

MODELS=(
  "$QWEN06|Qwen3-0.6B|Qwen3-0.6B"
  "$QWEN4B|Qwen3-4B|Qwen3-4B"
  "$QWEN8B|Qwen3-8B|Qwen3-8B"
  "$GEMMA|gemma-3-4b-it|gemma-3-4b-it"
  "$LLAMA|Meta-Llama-3.1-8B-Instruct|Meta-Llama-3.1-8B-Instruct"
)

cd "$TOMTEST"
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERR] 数据集目录不存在: $DATASET_ROOT"
  exit 1
fi

wait_vllm_port() {
  local port="$1"
  local tag="$2"
  local n=0
  echo "[INFO] [$tag] 等待 vLLM 监听 :${port} …"
  until curl -s -o /dev/null --connect-timeout 2 "http://127.0.0.1:${port}/v1/models" 2>/dev/null; do
    sleep 3
    n=$((n + 1))
    if [[ "$n" -gt 120 ]]; then
      echo "[ERR] [$tag] vLLM 启动超时（port=${port}）。"
      return 1
    fi
  done
  echo "[INFO] [$tag] vLLM 已就绪。"
}

for item in "${MODELS[@]}"; do
  IFS='|' read -r model_path served_name model_tag <<< "$item"
  if [[ ! -d "$model_path" ]]; then
    echo "[WARN] 跳过（模型路径不存在）: $model_tag -> $model_path"
    continue
  fi

  model_root="$TOMTEST/result/${model_tag}"
  result_dir="${model_root}/${DATASET_NAME}"
  serve_log="${model_root}/vllm_serve.log"
  run_log="${result_dir}/run.log"
  api_local="http://127.0.0.1:${PORT}/v1"
  mkdir -p "$result_dir"

  echo "========== [$model_tag] DATASET=${DATASET_NAME} GPU=${GPU_ID} PORT=${PORT} =========="
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU_ID" vllm serve "$model_path" \
    --port "$PORT" \
    --served-model-name "$served_name" \
    $VLLM_EXTRA \
    >"$serve_log" 2>&1 &
  VLLM_PID=$!

  cleanup_one() {
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  }
  trap cleanup_one EXIT INT TERM

  wait_vllm_port "$PORT" "$model_tag"

  cot_flag=()
  if [[ "$V2_COT" == "1" ]]; then
    cot_flag+=(--v2-cot)
  fi

  # shellcheck disable=SC2086
  python run.py \
    --dataset-root "$DATASET_ROOT" \
    --result-dir "$result_dir" \
    --dataset-filter "$DATASET_NAME" \
    --split-filter test \
    --backend api \
    --model "$served_name" \
    --model-tag "$model_tag" \
    --api-url "$api_local" \
    --api-key not-needed \
    --v2-language "$V2_LANGUAGE" \
    --v2-user-language "$V2_USER_LANGUAGE" \
    --v2-shuffle-votes "$V2_SHUFFLE_VOTES" \
    --v2-seed "$V2_SEED" \
    "${cot_flag[@]}" \
    $EXTRA_RUN_ARGS \
    >"$run_log" 2>&1

  cleanup_one
  trap - EXIT INT TERM
  echo "[DONE] [$model_tag] 完成 -> $result_dir"
done

echo "[DONE] ToMBench 5 模型完成（已排除 deepseek 系列）。"
