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

"""CPU regression tests for the repository-local Muon optimizer."""

import torch

from nemo_automodel.components.models.gpt2 import GPT2LMHeadModel
from nemo_automodel.components.optim.muon import Muon
from nemo_automodel.components.optim.optimizer import LocalMuonConfig, build_optimizer_config


def test_local_muon_routes_embedding_and_norm_to_adamw() -> None:
    """Only transformer linear matrices receive Newton-Schulz updates."""
    model = GPT2LMHeadModel(vocab_size=32, n_positions=4, n_embd=8, n_layer=1, n_head=2)
    config = build_optimizer_config(Muon, {"lr": 0.02, "scalar_lr": 0.001, "embedding_lr": 0.001})

    assert isinstance(config, LocalMuonConfig)
    optimizer = config.build(model)[0]
    algorithm_by_parameter = {
        id(parameter): group["algorithm"] for group in optimizer.param_groups for parameter in group["params"]
    }

    assert algorithm_by_parameter[id(model.wte.weight)] == "adamw"
    assert algorithm_by_parameter[id(model.ln_f.weight)] == "adamw"
    assert algorithm_by_parameter[id(model.h[0].attn.qkv_proj.weight)] == "muon"

    logits = model(torch.tensor([[1, 2, 3, 4]]))
    logits.sum().backward()
    embedding_before = model.wte.weight.detach().clone()
    optimizer.step()

    assert not torch.equal(model.wte.weight, embedding_before)
    assert optimizer.state[model.wte.weight]["exp_avg"].dtype is torch.float32
