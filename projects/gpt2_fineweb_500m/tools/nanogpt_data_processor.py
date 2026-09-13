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

"""Create a simple streaming FineWeb binary dataset for the GPT-2 recipe.

The processor intentionally uses one Python process. It opens the Hugging Face
stream, tokenizes one document at a time, and writes immediately to the binary
file. This is slower than a multi-process pipeline, but it has bounded memory,
exact token-budget handling, and no worker or queue lifecycle to deadlock.
"""

import argparse
import json
import logging
import os
import time
from typing import Any

import numpy as np
from transformers import PreTrainedTokenizerBase

try:
    from nemo_automodel.components.datasets.llm.nanogpt_dataset import HEADER_SIZE, MAGIC, VERSION
except ImportError:
    logging.warning("nemo_automodel is not installed; using local dataset constants")
    HEADER_SIZE = 256
    MAGIC = 2788_95051
    VERSION = 1


logger = logging.getLogger(__name__)


class _parse_tokens_arg(int):
    """Parse human-friendly token counts such as ``500M`` or ``1B``."""

    _UNIT_MULTIPLIER = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}

    def __new__(cls, value: str | int) -> "_parse_tokens_arg":
        if isinstance(value, int):
            return super().__new__(cls, value)
        val = value.strip()
        if val.isdigit():
            return super().__new__(cls, int(val))
        import re

        match = re.fullmatch(r"(?i)(\d+(?:\.\d+)?)\s*([KMB])", val)
        if match:
            return super().__new__(cls, int(float(match.group(1)) * cls._UNIT_MULTIPLIER[match.group(2).upper()]))
        raise argparse.ArgumentTypeError(f"Could not parse token count {value!r}; expected an integer or K/M/B value")

    def __repr__(self) -> str:
        value = int(self)
        for suffix, multiplier in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
            if value >= multiplier and value % multiplier == 0:
                return f"{value // multiplier}{suffix}"
        return str(value)


def make_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for FineWeb preprocessing."""
    parser = argparse.ArgumentParser(description="Stream, tokenize, and write a FineWeb binary dataset")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb", help="Hugging Face dataset identifier")
    parser.add_argument("--output-dir", default=None, help="Output directory without the max-token suffix")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--set-name", default="sample-10BT", help="FineWeb configuration name")
    parser.add_argument(
        "-m",
        "--max_tokens",
        "--max-tokens",
        type=_parse_tokens_arg,
        default=2**32,
        help="Stop after this many tokens; accepts values such as 500M or 1B",
    )
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer model ID")
    parser.add_argument("--data-cache-dir", default=None, help="Hugging Face dataset cache directory")
    parser.add_argument(
        "--max-length",
        type=int,
        default=32768,
        help="Maximum tokens per document, including the inserted BOS token",
    )
    parser.add_argument(
        "--local-parquet-dir",
        default=None,
        help="Optional local parquet directory for offline processing",
    )
    return parser


class BinaryDataWriter:
    """Write token IDs and BOS offsets to the NanoGPT binary format."""

    def __init__(self, filename: str, bos_token_id: int, vocab_size: int) -> None:
        """Initialize a writer for ``filename``.

        Args:
            filename: Output ``.bin`` path.
            bos_token_id: Token ID whose positions are written to ``.bos.idx``.
            vocab_size: Vocabulary size used to select uint16 or uint32 storage.
        """
        self.filename = filename
        self.bos_token_id = bos_token_id
        if vocab_size < 2**16:
            self.dtype = np.dtype(np.uint16)
        elif vocab_size < 2**32:
            self.dtype = np.dtype(np.uint32)
        else:
            raise ValueError(f"Vocabulary size {vocab_size} is too large for uint32 storage")
        logger.info("Using %s for vocabulary size %s", self.dtype.name, vocab_size)

        self.header = np.zeros(HEADER_SIZE, dtype=np.int32)
        self.header[0] = MAGIC
        self.header[1] = VERSION
        self.header[3] = self.dtype.itemsize
        self.bin_fp = None
        self.idx_fp = None
        self.bytes_written = 0

    def _write_header(self) -> tuple[Any, Any]:
        """Open output files and write the placeholder header."""
        bin_fp = open(self.filename, "wb")
        idx_fp = open(self.filename.replace(".bin", ".bos.idx"), "wb")
        bin_fp.write(self.header.tobytes())
        return bin_fp, idx_fp

    def write(self, tokens: np.ndarray | list[int], *, max_tokens: int | None = None) -> int:
        """Append one document and enforce the total token budget.

        Args:
            tokens: One-dimensional token IDs.
            max_tokens: Inclusive total dataset budget, if any.

        Returns:
            Number of tokens written from this document.
        """
        if self.bin_fp is None:
            self.bin_fp, self.idx_fp = self._write_header()
        tokens = np.asarray(tokens)
        if tokens.ndim != 1:
            raise ValueError(f"tokens must be one-dimensional, got shape {tokens.shape}")
        token_limit = np.iinfo(self.dtype).max
        if tokens.size and (tokens.min() < 0 or tokens.max() > token_limit):
            raise ValueError(f"token IDs must be in [0, {token_limit}] for {self.dtype.name}")
        if max_tokens is not None:
            remaining = max_tokens - self.items_written
            if remaining <= 0:
                return 0
            tokens = tokens[:remaining]
        if self.items_written + tokens.size > np.iinfo(np.int32).max:
            raise ValueError("dataset contains too many tokens for the int32 header")

        tokens = tokens.astype(self.dtype, copy=False)
        token_start = self.items_written
        self.bin_fp.write(tokens.tobytes())
        bos_positions = token_start + np.flatnonzero(tokens == self.bos_token_id)
        self.idx_fp.write(bos_positions.astype(np.int32, copy=False).tobytes())
        self.bytes_written += tokens.nbytes
        return int(tokens.size)

    @property
    def items_written(self) -> int:
        """Return the number of token IDs written so far."""
        return self.bytes_written // self.dtype.itemsize

    def close(self) -> None:
        """Finalize the token count in the header and close files."""
        if self.bin_fp is not None:
            self.header[2] = self.items_written
            self.bin_fp.seek(0)
            self.bin_fp.write(self.header.tobytes())
            self.bin_fp.close()
            self.bin_fp = None
        if self.idx_fp is not None:
            self.idx_fp.close()
            self.idx_fp = None

    def __enter__(self) -> "BinaryDataWriter":
        """Return this writer for a context manager."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close files even when tokenization fails."""
        self.close()


def _load_dataset(
    dataset_name: str,
    set_name: str,
    split: str,
    cache_dir: str,
    local_parquet_dir: str | None = None,
) -> Any:
    """Open the requested streaming dataset.

    Args:
        dataset_name: Hugging Face dataset identifier.
        set_name: Dataset configuration name.
        split: Dataset split.
        cache_dir: Local cache directory.
        local_parquet_dir: Optional local parquet directory.

    Returns:
        An iterable dataset containing text records.
    """
    from datasets import load_dataset

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if local_parquet_dir is not None:
        import glob

        parquet_files = sorted(glob.glob(os.path.join(local_parquet_dir, "**", "*.parquet"), recursive=True))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found under {local_parquet_dir}")
        logger.info("Loading %d local parquet files", len(parquet_files))
        return load_dataset(
            "parquet",
            data_files=parquet_files,
            split=split,
            streaming=True,
            cache_dir=cache_dir,
            token=hf_token,
        )

    logger.info(
        "Opening %s/%s:%s with streaming=True (HF_TOKEN=%s)",
        dataset_name,
        set_name,
        split,
        "set" if hf_token else "unset",
    )
    return load_dataset(
        dataset_name,
        name=set_name,
        split=split,
        streaming=True,
        cache_dir=cache_dir,
        token=hf_token,
    )


def tokenize_document(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_length: int,
    bos_token_id: int,
) -> list[int]:
    """Tokenize one document with a strict inclusive length limit.

    Args:
        tokenizer: Tokenizer that maps text to vocabulary IDs.
        text: Document text to tokenize.
        max_length: Maximum output length, including the BOS token.
        bos_token_id: Token ID prepended to every document.

    Returns:
        Token IDs with shape ``[tokens]`` and length at most ``max_length``.
    """
    if max_length < 1:
        raise ValueError(f"max_length must be positive, got {max_length}")
    tokens = tokenizer.encode(text, max_length=max_length - 1, truncation=True, add_special_tokens=False)
    return [bos_token_id, *tokens[: max_length - 1]]


def main(args: argparse.Namespace) -> None:
    """Stream, tokenize, and write a bounded binary dataset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Arguments: %s", json.dumps(vars(args), indent=2, default=str))

    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), args.dataset.split("/")[-1])
    output_dir = f"{output_dir}_max_tokens_{args.max_tokens}"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, default=str)
    logger.info("Writing to %s", output_dir)

    cache_dir = args.data_cache_dir or os.path.join(output_dir, "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, token=hf_token)
    if tokenizer.bos_token_id is None:
        raise ValueError(f"Tokenizer {args.tokenizer!r} has no bos_token_id")
    dataset_iter = _load_dataset(args.dataset, args.set_name, args.split, cache_dir, args.local_parquet_dir)
    writer = BinaryDataWriter(
        os.path.join(output_dir, "dataset.bin"),
        bos_token_id=tokenizer.bos_token_id,
        vocab_size=len(tokenizer),
    )

    started_at = time.monotonic()
    last_progress_at = started_at
    documents = 0
    with writer:
        for sample in dataset_iter:
            text = sample.get("text")
            if not isinstance(text, str) or not text:
                continue
            tokens = tokenize_document(tokenizer, text, args.max_length, tokenizer.bos_token_id)
            writer.write(tokens, max_tokens=args.max_tokens)
            documents += 1
            if writer.items_written >= args.max_tokens:
                break
            now = time.monotonic()
            if now - last_progress_at >= 10:
                elapsed = max(now - started_at, 1e-6)
                logger.info(
                    "Processed %d documents, wrote %d/%d tokens (%.0f tokens/s)",
                    documents,
                    writer.items_written,
                    int(args.max_tokens),
                    writer.items_written / elapsed,
                )
                last_progress_at = now

    elapsed = max(time.monotonic() - started_at, 1e-6)
    logger.info(
        "Dataset created: %s/dataset.bin (%d tokens from %d documents, %.0f tokens/s)",
        output_dir,
        writer.items_written,
        documents,
        writer.items_written / elapsed,
    )


if __name__ == "__main__":
    main(make_parser().parse_args())
