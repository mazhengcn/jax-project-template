#!/bin/bash
# Copyright 2022 Zheng Ma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e

DATASET_NAME=${1:-""}
RESTORE_DIR=${2:-""}
EVAL_CKPT_DIR=${3:-"ckpts/eval_ckpts"}
CUDA_DEVICES=${4:-""}

if [ -n "${CUDA_DEVICES}" ]; then
	export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
	DEVICES=($(tr "," " " <<< "${CUDA_DEVICES}"))
	BATCH_SIZE=${#DEVICES[@]}
else
	BATCH_SIZE="$(nvidia-smi --list-gpus | wc -l)"
fi

python project_name/train.py \
	--config=project_name/config.py \
	--config.experiment_kwargs.config.dataset.name=project_name/${DATASET_NAME} \
	--config.experiment_kwargs.config.evaluation.batch_size=${BATCH_SIZE} \
	--config.checkpoint_dir="${EVAL_CKPT_DIR}" \
	--config.restore_dir="${RESTORE_DIR}" \
	--config.one_off_evaluate="true" \
	--jaxline_mode="eval"
