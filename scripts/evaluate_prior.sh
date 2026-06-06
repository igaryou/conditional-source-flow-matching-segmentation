#!/bin/bash
set -e

PYTHON=${PYTHON:-/home/igarashi_25/playground_2/CSDFM/.venv/bin/python}
CONFIG=${CONFIG:-configs/cityscapes_csfm_128x256.json}
DATA_ROOT=${DATA_ROOT:-/home/igarashi_25/datasets/cityscapes}
CKPT_PATH=${CKPT_PATH:-outputs/csfm_cityscapes_128x256/model_final.pth}
SAVE_DIR=${SAVE_DIR:-results/csfm_cityscapes_128x256/prior}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
"${PYTHON}" src/evaluate_prior.py \
  --config "${CONFIG}" \
  --data_root "${DATA_ROOT}" \
  --ckpt_path "${CKPT_PATH}" \
  --save_dir "${SAVE_DIR}" \
  --split val \
  --num_visualize 16 \
  --sample_seed 0
