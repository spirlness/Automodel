#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

set -euo pipefail

DATASET_NAME="${1:-HuggingFaceFW/fineweb}"
SET_NAME="${2:-sample-10BT}"
MAX_TOKENS="${3:-500M}"
OUTPUT_DIR="${4:-projects/gpt2_fineweb_500m/data/fineweb_500M}"

echo "=== [1/1] Streaming and tokenizing FineWeb dataset (${MAX_TOKENS} tokens) ==="
uv run python projects/gpt2_fineweb_500m/tools/nanogpt_data_processor.py \
  --dataset "${DATASET_NAME}" \
  --set-name "${SET_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-tokens "${MAX_TOKENS}" \
  --max-length 1024

echo "=== Data preparation completed successfully ==="
