#!/usr/bin/env bash
set -euo pipefail

# Edit these values before running.
GPU_ID="3"
RUN_NAME="修改了动态prompt没有接入的情况，修改了stage1a和b没有正确配置的情况，跳过了stage2b因为没有给出函数"
CONFIG_FILE="configs/ours/cctv_ir_cctv_rgb.yml"
TRAIN_SCRIPT="train_uniprompt.py"
PYTHON_BIN="python"

# Optional extra config overrides.
# Add items as pairs: "KEY" "VALUE"
EXTRA_OPTS=(
  # "OUTPUT_DIR" "/data1/hewenke/mp-reid/log"
  # "SOLVER.STAGE2.MAX_EPOCHS" "10"
  # "SOLVER.STAGE1A.MAX_EPOCHS" "5"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "GPU_ID=${GPU_ID}"
echo "RUN_NAME=${RUN_NAME}"
echo "CONFIG_FILE=${CONFIG_FILE}"
echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --config_file "${CONFIG_FILE}" \
  RUN_NAME "${RUN_NAME}" \
  "${EXTRA_OPTS[@]}"
