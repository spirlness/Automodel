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

"""CPU regression tests for the repository-local GPT-2 builder."""

import pytest
import torch

from nemo_automodel.components.models.gpt2 import GPT2LMHeadModel, build_gpt2_model


def test_local_gpt2_forward_and_sequence_limit() -> None:
    """The local model produces logits and rejects an overlong sequence clearly."""
    model = GPT2LMHeadModel(vocab_size=32, n_positions=4, n_embd=8, n_layer=1, n_head=2).bfloat16()
    logits = model(torch.tensor([[1, 2, 3, 4]]))

    assert logits.shape == (1, 4, 32)
    assert logits.dtype is torch.bfloat16
    fused_loss_output = model(torch.tensor([[1, 2, 3, 4]]), logits_to_keep=1)
    assert fused_loss_output["hidden_states"].shape == (1, 4, 8)
    assert fused_loss_output["hidden_states"].dtype is torch.bfloat16
    with pytest.raises(ValueError, match="Sequence length 5 exceeds the configured context length 4"):
        model(torch.tensor([[1, 2, 3, 4, 5]]))


def test_build_local_gpt2_with_bfloat16_parameters() -> None:
    """The model builder accepts the recipe's explicit low-precision dtype."""
    model = build_gpt2_model(vocab_size=32, n_positions=4, n_embd=8, n_layer=1, n_head=2, torch_dtype="bfloat16")

    assert model.wte.weight.dtype is torch.bfloat16
    assert model.lm_head.weight is model.wte.weight
    assert model.h[0].attn.freqs_real.dtype is torch.float32
    assert model.h[0].attn.freqs_real.device == model.wte.weight.device

    fp16_model = build_gpt2_model(vocab_size=32, n_positions=4, n_embd=8, n_layer=1, n_head=2, torch_dtype="float16")
    assert fp16_model.h[0].attn.freqs_real.dtype is torch.float32
    assert fp16_model.h[0].attn.freqs_real.device == fp16_model.wte.weight.device


def test_local_gpt2_requires_sdpa() -> None:
    """The builder does not silently claim to select a flash-attn backend."""
    with pytest.raises(ValueError, match="supports only attn_implementation='sdpa'"):
        build_gpt2_model(
            vocab_size=32, n_positions=4, n_embd=8, n_layer=1, n_head=2, attn_implementation="flash_attention_2"
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_local_gpt2_gpu_training_step() -> None:
    """The local model and grouped Muon optimizer complete one bf16 CUDA step."""
    from nemo_automodel.components.optim.muon import Muon
    from nemo_automodel.components.optim.optimizer import build_optimizer_config

    model = GPT2LMHeadModel(vocab_size=32, n_positions=4, n_embd=8, n_layer=1, n_head=2).cuda().bfloat16()
    optimizer = build_optimizer_config(Muon, {"lr": 0.01, "scalar_lr": 0.001, "embedding_lr": 0.001}).build(model)[0]
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    loss = model(input_ids).square().mean()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
