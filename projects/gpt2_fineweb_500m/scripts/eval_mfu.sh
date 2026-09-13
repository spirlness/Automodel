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

STEPS="${1:-100}"
CONFIG_PATH="projects/gpt2_fineweb_500m/config/gpt2_fineweb_500m.yaml"

echo "=== Running ${STEPS}-step MFU and Throughput Benchmark ==="
uv run automodel "${CONFIG_PATH}" \
  --nproc-per-node 1 \
  --step_scheduler.local_batch_size 2 \
  --step_scheduler.global_batch_size 16 \
  --step_scheduler.max_steps "${STEPS}" \
  --lr_scheduler.lr_warmup_steps 10 \
  --checkpoint.enabled false

echo "=== MFU benchmark finished ==="
