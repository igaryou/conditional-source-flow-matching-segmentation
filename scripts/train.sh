#!/bin/bash
set -e

PYTHON=${PYTHON:-/home/igarashi_25/playground_2/CSDFM/.venv/bin/python}
CONFIG=${CONFIG:-configs/cityscapes_csfm_128x256.json}
DATA_ROOT=${DATA_ROOT:-/home/igarashi_25/datasets/cityscapes}
RESULT_DIR=${RESULT_DIR:-outputs/csfm_cityscapes_128x256}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
"${PYTHON}" src/train.py \
  --config "${CONFIG}" \
  --data_root "${DATA_ROOT}" \
  --result_dir "${RESULT_DIR}" \
  --amp \
  --amp_dtype bf16 \
  --use_checkpoint
