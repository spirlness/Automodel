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

CONFIG_PATH="projects/gpt2_fineweb_500m/config/gpt2_fineweb_500m.yaml"
DATA_PATH="projects/gpt2_fineweb_500m/data/fineweb_500M_max_tokens_500M/dataset.bin"

echo "=== Running 2-step smoke test ==="
uv run automodel "${CONFIG_PATH}" \
  --nproc-per-node 1 \
  --dataset.file_pattern="${DATA_PATH}" \
  --step_scheduler.max_steps=2 \
  --checkpoint.enabled=false

echo "=== Smoke test completed successfully ==="
