# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Generate the minimal Kaggle notebook for GPT-2 FineWeb pretraining."""

import json
from pathlib import Path

REPOSITORY_URL = "https://github.com/spirlness/Automodel.git"
SOURCE_COMMIT = "82b6ab364"
RECIPE_PATH = "projects/gpt2_fineweb_500m/config/gpt2_fineweb_t4x2.yaml"
RECIPE_ABS_PATH = f"/kaggle/working/Automodel/{RECIPE_PATH}"
DATA_DIR = "/kaggle/working/fineweb_1B"
SMOKE_DATA_DIR = "/kaggle/working/fineweb_smoke"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
MAX_TOKENS = "1B"
SMOKE_MAX_TOKENS = "1M"
GLOBAL_BATCH_SIZE = 8
LOCAL_BATCH_SIZE = 4
MAX_STEPS = 1_000_000_000 // (GLOBAL_BATCH_SIZE * 1024)
CHECKPOINT_INTERVAL = 10_000


def _code_cell(cell_id: str, source: str) -> dict[str, object]:
    """Create a stable notebook code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> dict[str, object]:
    """Build the minimal, smoke-tested Kaggle training notebook."""
    return {
        "cells": [
            _code_cell(
                "gpu-info",
                """# The recipe is configured for two Kaggle T4 GPUs.
!nvidia-smi
import torch

gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
print(f"GPU count: {len(gpu_names)}")
print("GPUs:", gpu_names)
if len(gpu_names) != 2 or any("T4" not in name for name in gpu_names):
    raise RuntimeError(f"Select Kaggle GPU T4 x2; allocated hardware is {gpu_names}.")
""",
            ),
            _code_cell(
                "environment",
                f"""# Use the exact repository commit that generated this notebook.
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
import subprocess

os.environ["PATH"] = f"/root/.local/bin:{{os.environ['PATH']}}"
!git clone --filter=blob:none --no-checkout {REPOSITORY_URL} Automodel
os.chdir("/kaggle/working/Automodel")
!git fetch --depth 1 origin {SOURCE_COMMIT}
!git checkout --detach {SOURCE_COMMIT}
checked_out_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if not checked_out_commit.startswith("{SOURCE_COMMIT}"):
    raise RuntimeError(f"Expected source commit {SOURCE_COMMIT}, got {{checked_out_commit}}")
print(f"Using source commit {{checked_out_commit}}")
!uv sync --locked --no-default-groups --inexact

# FineWeb is public. If the Kaggle environment provides HF_TOKEN, the
# processor will use it; an HF Secret is not required for this run.
if os.environ.get("HF_TOKEN"):
    print("HF_TOKEN is available")
else:
    print("HF_TOKEN is not set; using the public FineWeb dataset")
""",
            ),
            _code_cell(
                "smoke-preprocess",
                f"""# Verify Hub access, tokenization, and exact binary writing first.
!uv run python projects/gpt2_fineweb_500m/tools/nanogpt_data_processor.py \\
  --dataset HuggingFaceFW/fineweb \\
  --set-name sample-10BT \\
  --output-dir {SMOKE_DATA_DIR} \\
  --max-tokens {SMOKE_MAX_TOKENS} \\
  --max-length 1024
""",
            ),
            _code_cell(
                "smoke-config",
                f"""# Construct the exact model and loss from the Kaggle YAML before torchrun.
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

cfg = parse_args_and_load_config("{RECIPE_ABS_PATH}")
model = cfg.model.instantiate()
loss = cfg.loss_fn.instantiate()
assert model.__class__.__name__ == "GPT2LMHeadModel"
assert loss.__class__.__name__ == "MaskedCrossEntropy"
assert cfg.optimizer._target_.__module__ == "torch.optim"
assert cfg.optimizer._target_.__name__ == "AdamW"
assert cfg.distributed.strategy == "ddp"
print(f"Config OK: model={{model.__class__.__name__}}, loss={{loss.__class__.__name__}}, strategy={{cfg.distributed.strategy}}")
del model, loss, cfg
""",
            ),
            _code_cell(
                "smoke-train",
                f"""# Run one baseline DDP + AdamW training step on the smoke dataset.
!uv run automodel {RECIPE_ABS_PATH} \\
  --nproc-per-node 2 \\
  --dataset.file_pattern={SMOKE_DATA_DIR}_max_tokens_{SMOKE_MAX_TOKENS}/dataset.bin \\
  --step_scheduler.global_batch_size={GLOBAL_BATCH_SIZE} \\
  --step_scheduler.local_batch_size={LOCAL_BATCH_SIZE} \\
  --step_scheduler.max_steps=1 \\
  --step_scheduler.ckpt_every_steps=1000000 \\
  --step_scheduler.val_every_steps=1000000 \\
  --step_scheduler.save_checkpoint_every_epoch=false \\
  --checkpoint.enabled=false
""",
            ),
            _code_cell(
                "full-train",
                f"""# The smoke test passed, so build the 1B-token dataset and train.
!uv run python projects/gpt2_fineweb_500m/tools/nanogpt_data_processor.py \\
  --dataset HuggingFaceFW/fineweb \\
  --set-name sample-10BT \\
  --output-dir {DATA_DIR} \\
  --max-tokens {MAX_TOKENS} \\
  --max-length 1024
!uv run automodel {RECIPE_ABS_PATH} \\
  --nproc-per-node 2 \\
  --dataset.file_pattern={DATA_DIR}_max_tokens_{MAX_TOKENS}/dataset.bin \\
  --step_scheduler.global_batch_size={GLOBAL_BATCH_SIZE} \\
  --step_scheduler.local_batch_size={LOCAL_BATCH_SIZE} \\
  --step_scheduler.max_steps={MAX_STEPS} \\
  --step_scheduler.ckpt_every_steps={CHECKPOINT_INTERVAL} \\
  --step_scheduler.val_every_steps=1000000 \\
  --step_scheduler.save_checkpoint_every_epoch=false \\
  --checkpoint.checkpoint_dir={CHECKPOINT_DIR} \\
  --checkpoint.max_recent_checkpoints=3
""",
            ),
        ],
        "metadata": {
            "accelerator": "nvidiaTeslaT4",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    """Write the canonical Kaggle notebook next to this generator."""
    output_path = Path(__file__).with_name("train_gpt2_t4x2.ipynb")
    output_path.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
