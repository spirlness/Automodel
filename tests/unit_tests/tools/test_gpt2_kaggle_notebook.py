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

"""Tests for the Kaggle notebook generator of the GPT-2 FineWeb project."""

import importlib.util
from pathlib import Path


def _load_generator_module():
    path = Path(__file__).parents[3] / "projects/gpt2_fineweb_500m/kaggle_deployment/generate_nb.py"
    spec = importlib.util.spec_from_file_location("gpt2_kaggle_notebook", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_kaggle_notebook_uses_the_project_recipe_without_source_injection() -> None:
    """The generated notebook checks out this branch and invokes project-owned tooling."""
    generator = _load_generator_module()
    source = "\n".join("".join(cell["source"]) for cell in generator.build_notebook()["cells"])

    assert generator.REPOSITORY_URL in source
    assert generator.BRANCH in source
    assert generator.RECIPE_PATH in source
    assert "nanogpt_data_processor.py" in source
    assert "--max-tokens 1B" in source
    assert "--nproc-per-node 2" in source
    assert "--step_scheduler.global_batch_size=32" in source
    assert "--step_scheduler.local_batch_size=4" in source
    assert "--step_scheduler.max_steps=30517" in source
    assert "--checkpoint.max_recent_checkpoints=3" in source
    assert "PYTORCH_ALLOC_CONF=expandable_segments:True" in source
    assert generator.MAX_STEPS == 1_000_000_000 // (generator.GLOBAL_BATCH_SIZE * 1024)
    assert generator.GLOBAL_BATCH_SIZE // (generator.LOCAL_BATCH_SIZE * 2) == 4
    assert all(isinstance(cell["id"], str) and cell["id"] for cell in generator.build_notebook()["cells"])
    assert 'with open("nemo_automodel/components/models/gpt2.py"' not in source
    assert 'with open("nemo_automodel/components/optim/muon.py"' not in source


def test_checked_in_notebook_matches_generator() -> None:
    """The committed notebook is regenerated whenever the generator changes."""
    import json

    generator = _load_generator_module()
    notebook_path = Path(__file__).parents[3] / "projects/gpt2_fineweb_500m/kaggle_deployment/train_gpt2_t4x2.ipynb"
    assert json.loads(notebook_path.read_text(encoding="utf-8")) == generator.build_notebook()
