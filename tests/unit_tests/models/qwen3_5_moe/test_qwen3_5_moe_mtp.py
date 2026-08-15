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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5_moe")

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForConditionalGeneration


def _tiny_vl_config(**text_kwargs):
    text_config = Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        layer_types=["full_attention"],
        **text_kwargs,
    )
    vision_config = dict(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=16,
        num_position_embeddings=8,
    )
    return Qwen3_5MoeConfig(text_config=text_config.to_dict(), vision_config=vision_config)


def _backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
    )


class TestQwen3_5MoeMTP:
    def test_declares_mtp_cp_support(self):
        assert Qwen3_5MoeForConditionalGeneration.ModelCapabilities().supports_mtp_cp is True

    def test_moe_mtp_builds_full_attention_moe_block(self):
        model = Qwen3_5MoeForConditionalGeneration(_tiny_vl_config(mtp_num_hidden_layers=1), backend=_backend())

        assert model.mtp is not None
        mtp_layer = model.mtp.layers[0]
        assert mtp_layer.layer_type == "full_attention"
        assert hasattr(mtp_layer, "self_attn")
        assert hasattr(mtp_layer, "eh_proj")
        assert hasattr(mtp_layer, "final_layernorm")

        mtp_keys = [key for key in model.state_dict() if key.startswith("mtp.")]
        assert "mtp.layers.0.mlp.gate.weight" in mtp_keys
        assert "mtp.layers.0.mlp.experts.gate_and_up_projs" in mtp_keys
        assert "mtp.layers.0.mlp.shared_experts.gate_proj.weight" in mtp_keys

    def test_moe_mtp_emits_with_media_token_ids(self, monkeypatch):
        cfg = _tiny_vl_config(mtp_num_hidden_layers=1)
        cfg.image_token_id = 1
        cfg.vision_start_token_id = 2
        model = Qwen3_5MoeForConditionalGeneration(cfg, backend=_backend())
        model.train()

        hidden_states = torch.randn(1, 4, cfg.text_config.hidden_size, dtype=model.lm_head.weight.dtype)

        def fake_model_forward(**kwargs):
            assert "pixel_values" in kwargs
            return SimpleNamespace(last_hidden_state=hidden_states)

        def fake_mtp_forward(hidden_states, **kwargs):
            assert "pixel_values" not in kwargs
            assert "image_grid_thw" not in kwargs
            return [hidden_states]

        monkeypatch.setattr(model.model, "forward", fake_model_forward)
        monkeypatch.setattr(model.mtp, "forward", fake_mtp_forward)

        out = model(
            input_ids=torch.tensor([[3, 2, 1, 4]], dtype=torch.long),
            pixel_values=torch.empty(0),
            image_grid_thw=torch.empty((0, 3), dtype=torch.long),
        )

        assert out.logits.shape == (1, 4, cfg.text_config.vocab_size)
        assert out.mtp_per_depth_h is not None
        assert len(out.mtp_per_depth_h) == 1
        assert out.mtp_per_depth_h[0].shape == (1, 4, cfg.text_config.hidden_size)

    @pytest.mark.parametrize(
        ("uses_blockdiag_cp", "expected_main", "expected_mtp", "valid_mask"),
        [
            (False, [0, 1, 6, 7], [1, 500, 7, 0], [True, True, True, False]),
            (True, [0, 1, 500, 3], [1, 500, 3, 4], [True, True, True, True]),
        ],
    )
    def test_cp_mtp_uses_globally_rolled_fused_embeddings(
        self,
        monkeypatch,
        uses_blockdiag_cp,
        expected_main,
        expected_mtp,
        valid_mask,
    ):
        cfg = _tiny_vl_config(mtp_num_hidden_layers=1)
        model = Qwen3_5MoeForConditionalGeneration(cfg, backend=_backend())
        model.train()

        class _CPMesh:
            @staticmethod
            def size():
                return 2

            @staticmethod
            def get_local_rank():
                return 0

        model.cp_mesh = _CPMesh()
        hidden_size = cfg.text_config.hidden_size
        full_fused = torch.arange(8, dtype=model.lm_head.weight.dtype).view(1, 8, 1).expand(1, 8, hidden_size)
        full_fused = full_fused.clone()
        full_fused[:, 2] = 500  # stand in for a vision-spliced embedding
        captured = {}

        monkeypatch.setattr(model, "_embed_and_splice_for_cp", lambda input_ids, **kwargs: full_fused)
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.blockdiag_cp.current_blockdiag_cp_state",
            lambda: object() if uses_blockdiag_cp else None,
        )

        def fake_model_forward(**kwargs):
            captured["main_embed_inputs"] = kwargs["inputs_embeds"].detach().clone()
            return SimpleNamespace(last_hidden_state=torch.zeros_like(kwargs["inputs_embeds"]))

        def fake_mtp_forward(hidden_states, **kwargs):
            captured["mtp_embed_inputs"] = tuple(t.detach().clone() for t in kwargs["embed_inputs"])
            captured["mtp_position_ids"] = tuple(t.detach().clone() for t in kwargs["position_ids_per_depth"])
            return [hidden_states]

        monkeypatch.setattr(model.model, "forward", fake_model_forward)
        monkeypatch.setattr(model.mtp, "forward", fake_mtp_forward)

        local_positions = torch.arange(12).reshape(3, 1, 4)
        out = model(
            input_ids=torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]]),
            position_ids=local_positions,
            mtp_per_depth_input_ids=(torch.tensor([[11, 12, 17, 0]]),),
            mtp_per_depth_position_ids=(local_positions + 1,),
            mtp_per_depth_valid_masks=(torch.tensor([valid_mask]),),
        )

        assert out.mtp_per_depth_h is not None
        assert captured["main_embed_inputs"][0, :, 0].tolist() == expected_main
        assert captured["mtp_embed_inputs"][0][0, :, 0].tolist() == expected_mtp
        torch.testing.assert_close(captured["mtp_position_ids"][0], local_positions + 1)
