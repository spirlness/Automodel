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

"""Generate the Kaggle notebook that runs this repository's GPT-2 recipe."""

import json
from pathlib import Path

REPOSITORY_URL = "https://github.com/spirlness/Automodel.git"
SOURCE_COMMIT = "84552b22a"
RECIPE_PATH = "projects/gpt2_fineweb_500m/config/gpt2_fineweb_t4x2.yaml"
DATA_DIR = "/kaggle/working/fineweb_1B"
SMOKE_DATA_DIR = "/kaggle/working/fineweb_smoke"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
MAX_TOKENS = "1B"
SMOKE_MAX_TOKENS = "1M"
GLOBAL_BATCH_SIZE = 32
LOCAL_BATCH_SIZE = 4
MAX_STEPS = 30517
CHECKPOINT_INTERVAL = 10000


def _code_cell(cell_id: str, source: str) -> dict[str, object]:
    """Return a notebook code cell with a stable Notebook 4.5+ identifier."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> dict[str, object]:
    """Build a Kaggle notebook that uses the checked-out project implementation."""
    return {
        "cells": [
            _code_cell(
                "gpu-info",
                """# GPT-2 FineWeb pretraining using the reviewed repository implementation
!nvidia-smi
import torch

num_gpus = torch.cuda.device_count()
gpu_names = [torch.cuda.get_device_name(index) for index in range(num_gpus)]
print(f"GPU count: {num_gpus}")
for index, name in enumerate(gpu_names):
    print(f"GPU {index}: {name}")
if num_gpus != 2 or any("T4" not in name for name in gpu_names):
    raise RuntimeError(
        "This training requires two Tesla T4 GPUs. In Kaggle, select GPU T4 x2; "
        f"the allocated hardware is {gpu_names}."
    )
""",
            ),
            _code_cell(
                "environment",
                f"""# Install exactly the tested source commit.
!pip install -q uv
!git clone --filter=blob:none --no-checkout {REPOSITORY_URL} Automodel
%cd Automodel
!git fetch --depth 1 origin {SOURCE_COMMIT}
!git checkout --detach {SOURCE_COMMIT}
import subprocess
checked_out_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if not checked_out_commit.startswith("{SOURCE_COMMIT}"):
    raise RuntimeError(f"Expected source commit {SOURCE_COMMIT}, got {{checked_out_commit}}")
print(f"Using source commit {{checked_out_commit}}")
!uv sync --locked --no-default-groups --inexact

# Prefer the Kaggle Secret, but do not make a public FineWeb run depend on the
# Kaggle secret service being available. Never print the token.
from kaggle_secrets import UserSecretsClient
import os

hf_token = os.environ.get("HF_TOKEN")
try:
    secret_token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as exc:
    secret_token = None
    print(f"HF_TOKEN Kaggle Secret is unavailable ({{type(exc).__name__}}); continuing without it.")
if secret_token:
    hf_token = secret_token
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
if hf_token:
    from huggingface_hub import whoami
    whoami(token=hf_token)
    print("Loaded and validated HF_TOKEN.")
else:
    print("No HF_TOKEN available; FineWeb is public, continuing unauthenticated.")
""",
            ),
            _code_cell(
                "smoke-preprocess",
                f"""# First verify download, tokenization, and binary writing on a bounded sample.
!uv run python projects/gpt2_fineweb_500m/tools/nanogpt_data_processor.py \
  --dataset HuggingFaceFW/fineweb \
  --set-name sample-10BT \
  --output-dir {SMOKE_DATA_DIR} \
  --max-tokens {SMOKE_MAX_TOKENS} \
  --chunk-size 16 \
  --prefetch 4 \
  --num-workers 2
""",
            ),
            _code_cell(
                "smoke-train",
                f"""# Construct the exact model and loss from the Kaggle YAML before torchrun.
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

cfg = parse_args_and_load_config("{RECIPE_PATH}")
model = cfg.model.instantiate()
loss = cfg.loss_fn.instantiate()
assert model.__class__.__name__ == "GPT2LMHeadModel"
assert loss.__class__.__name__ == "MaskedCrossEntropy"
print(f"Config construction OK: model={{model.__class__.__name__}}, loss={{loss.__class__.__name__}}")
del model, loss, cfg
""",
            ),
            _code_cell(
                "smoke-train-run",
                f"""# Exercise two-rank FSDP, FP16, ordinary CE, backward, and optimizer update.
# This runs one full-size global training step using the dedicated Kaggle YAML.
!PYTORCH_ALLOC_CONF=expandable_segments:True uv run automodel {RECIPE_PATH} \
  --nproc-per-node 2 \
  --dataset.file_pattern={SMOKE_DATA_DIR}_max_tokens_{SMOKE_MAX_TOKENS}/dataset.bin \
  --step_scheduler.global_batch_size={GLOBAL_BATCH_SIZE} \
  --step_scheduler.local_batch_size={LOCAL_BATCH_SIZE} \
  --step_scheduler.max_steps=1 \
  --step_scheduler.ckpt_every_steps=1 \
  --step_scheduler.val_every_steps=1000000 \
  --step_scheduler.save_checkpoint_every_epoch=false \
  --checkpoint.enabled=false \
  --model.torch_dtype=float16 \
  --distributed.mp_policy.param_dtype=torch.float16 \
  --distributed.mp_policy.output_dtype=torch.float16
""",
            ),
            _code_cell(
                "full-train",
                f"""# Smoke test passed. Keep this False until its logs confirm both preprocessing
# and one optimizer update completed successfully. Then set it to True and run this cell.
RUN_FULL_TRAINING = False

if RUN_FULL_TRAINING:
    get_ipython().system(
        "PYTORCH_ALLOC_CONF=expandable_segments:True uv run python "
        "projects/gpt2_fineweb_500m/tools/nanogpt_data_processor.py "
        "--dataset HuggingFaceFW/fineweb --set-name sample-10BT "
        "--output-dir {DATA_DIR} --max-tokens {MAX_TOKENS}"
    )
    get_ipython().system(
        "PYTORCH_ALLOC_CONF=expandable_segments:True uv run automodel {RECIPE_PATH} "
        "--nproc-per-node 2 "
        "--dataset.file_pattern={DATA_DIR}_max_tokens_{MAX_TOKENS}/dataset.bin "
        "--step_scheduler.global_batch_size={GLOBAL_BATCH_SIZE} "
        "--step_scheduler.local_batch_size={LOCAL_BATCH_SIZE} "
        "--step_scheduler.max_steps={MAX_STEPS} "
        "--step_scheduler.ckpt_every_steps={CHECKPOINT_INTERVAL} "
        "--step_scheduler.save_checkpoint_every_epoch=false "
        "--checkpoint.checkpoint_dir={CHECKPOINT_DIR} "
        "--checkpoint.max_recent_checkpoints=3 "
        "--model.torch_dtype=float16 "
        "--distributed.mp_policy.param_dtype=torch.float16 "
        "--distributed.mp_policy.output_dtype=torch.float16"
    )
else:
    print("Full 1B-token training is disabled until the smoke test passes.")
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
