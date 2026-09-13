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


def test_tokenize_document_enforces_inclusive_length() -> None:
    """The document limit includes the inserted BOS token."""
    processor = _load_processor_module()

    class Tokenizer:
        def __init__(self) -> None:
            self.kwargs = None

        def encode(self, text: str, **kwargs) -> list[int]:
            self.kwargs = kwargs
            return [1, 2, 3]

    tokenizer = Tokenizer()
    tokens = processor.tokenize_document(tokenizer, "example", 3, bos_token_id=9)

    assert tokens == [9, 1, 2]
    assert tokenizer.kwargs == {"max_length": 2, "truncation": True, "add_special_tokens": False}


def test_tokenize_document_rejects_non_positive_length() -> None:
    """Invalid document limits fail before invoking the tokenizer."""
    processor = _load_processor_module()

    class Tokenizer:
        def encode(self, text: str, **kwargs) -> list[int]:
            raise AssertionError("encode must not be called")

    import pytest

    with pytest.raises(ValueError, match="max_length must be positive"):
        processor.tokenize_document(Tokenizer(), "example", 0, bos_token_id=9)


def test_main_writes_exact_budget_without_worker_processes(tmp_path: Path, monkeypatch) -> None:
    """The sequential path stops inside a document at the exact token budget."""
    processor = _load_processor_module()

    class Tokenizer:
        bos_token_id = 9

        def __len__(self) -> int:
            return 32

        def encode(self, text: str, **kwargs) -> list[int]:
            return [1, 2, 3]

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs) -> Tokenizer:
            return Tokenizer()

    monkeypatch.setattr("transformers.AutoTokenizer", AutoTokenizer)
    fake_datasets = types.SimpleNamespace(load_dataset=lambda *args, **kwargs: iter([{"text": "one"}, {"text": "two"}]))
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    args = processor.make_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "fineweb"),
            "--max-tokens",
            "5",
            "--max-length",
            "4",
        ]
    )
    processor.main(args)

    output = tmp_path / "fineweb_max_tokens_5" / "dataset.bin"
    header = np.fromfile(output, dtype=np.int32, count=processor.HEADER_SIZE)
    assert header[2] == 5
