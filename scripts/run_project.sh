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

DATA_DIR=${1:-""}
DATA_FILENAMES=${2:-""}
MODEL_DIR=${3:-""}
CUDA_DEVICES=${4:-""}

if [ -n "${CUDA_DEVICES}" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
fi

python run_project.py \
    --output_dir="results" \
    --data_dir="${DATA_DIR}" \
    --data_filenames="${DATA_FILENAMES}" \
    --model_dir="${MODEL_DIR}"
