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
BRANCH = "spirlness/feat/gpt2-fineweb-training"
RECIPE_PATH = "projects/gpt2_fineweb_500m/config/gpt2_fineweb_500m.yaml"
DATA_DIR = "/kaggle/working/fineweb"


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

num_gpus = max(1, torch.cuda.device_count())
print(f"GPU count: {num_gpus}")
for index in range(num_gpus):
    print(torch.cuda.get_device_name(index))
""",
            ),
            _code_cell(
                "environment",
                f"""# Install exactly the branch that contains this recipe.
!pip install -q uv
!git clone --depth 1 --branch {BRANCH} {REPOSITORY_URL} Automodel
%cd Automodel
!uv sync --locked --group dev --extra fa --inexact
""",
            ),
            _code_cell(
                "preprocess",
                f"""# Produce the binary dataset with the repository-owned preprocessor.
!uv run python projects/gpt2_fineweb_500m/tools/nanogpt_data_processor.py \\
  --dataset HuggingFaceFW/fineweb \\
  --set-name sample-10BT \\
  --output-dir {DATA_DIR} \\
  --max-tokens 500M
""",
            ),
            _code_cell(
                "train",
                f"""# Run the same recipe as the repository. T4 uses fp16 runtime overrides;
# all architecture, optimizer grouping, loss, and data-processing code stays identical.
import torch

num_gpus = max(1, torch.cuda.device_count())
global_batch_size = 32 if num_gpus >= 2 else 16
!uv run automodel {RECIPE_PATH} \\
  --nproc-per-node {{num_gpus}} \\
  --dataset.file_pattern={DATA_DIR}_max_tokens_500M/dataset.bin \\
  --step_scheduler.global_batch_size={{global_batch_size}} \\
  --step_scheduler.local_batch_size=4 \\
  --model.torch_dtype=float16 \\
  --distributed.mp_policy.param_dtype=torch.float16 \\
  --distributed.mp_policy.output_dtype=torch.float16
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
