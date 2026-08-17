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

"""Regression tests for the FineWeb binary-data preprocessing tool."""

import importlib.util
import sys
import types
from pathlib import Path
from queue import Queue

import numpy as np


def _load_processor_module():
    path = Path(__file__).parents[3] / "projects/gpt2_fineweb_500m/tools/nanogpt_data_processor.py"
    spec = importlib.util.spec_from_file_location("nanogpt_data_processor", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_writer_enforces_budget_finalizes_header_and_uses_token_offsets(tmp_path: Path) -> None:
    """The writer caps tokens exactly and stores BOS positions in token units."""
    processor = _load_processor_module()
    filename = tmp_path / "dataset.bin"
    with processor.BinaryDataWriter(str(filename), bos_token_id=9, vocab_size=32) as writer:
        assert writer.write([9, 1, 9, 2], max_tokens=3) == 3
        assert writer.write([9, 3], max_tokens=3) == 0

    header = np.fromfile(filename, dtype=np.int32, count=processor.HEADER_SIZE)
    bos_positions = np.fromfile(filename.with_suffix(".bos.idx"), dtype=np.int32)
    assert header[2] == 3
    assert bos_positions.tolist() == [0, 2]


def test_tokenize_chunk_enables_truncation() -> None:
    """The configured maximum length is passed to the tokenizer as a hard limit."""
    processor = _load_processor_module()

    class Tokenizer:
        bos_token_id = 9

        def __init__(self) -> None:
            self.kwargs = None

        def encode(self, text: str, **kwargs) -> list[int]:
            self.kwargs = kwargs
            return [1, 2]

    tokenizer = Tokenizer()
    processor._get_tokenizer = lambda _: (tokenizer, tokenizer.bos_token_id)
    processor.tokenize_chunk([{"text": "example"}], "test", 2)

    assert tokenizer.kwargs == {"max_length": 2, "truncation": True}


def test_dataset_reader_reports_loader_errors_instead_of_hanging() -> None:
    """A Hub failure must reach the parent process as an error message and done marker."""
    processor = _load_processor_module()

    def failing_load_dataset(*_: object, **__: object) -> object:
        raise RuntimeError("simulated Hub timeout")

    fake_datasets = types.SimpleNamespace(load_dataset=failing_load_dataset)
    original_datasets = sys.modules.get("datasets")
    sys.modules["datasets"] = fake_datasets
    messages: Queue = Queue()
    try:
        processor.dataset_reader(
            "HuggingFaceFW/fineweb",
            "sample-10BT",
            "train",
            "/tmp/cache",
            messages,
            2,
        )
    finally:
        if original_datasets is None:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = original_datasets

    kind, error = messages.get_nowait()
    assert kind == "error"
    assert "simulated Hub timeout" in error
    assert messages.get_nowait() == ("done", None)
