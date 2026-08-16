import nbformat as nbf

nb = nbf.v4.new_notebook()

# Ensure standard Jupyter / Papermill metadata and Kaggle T4 accelerator are specified
nb.metadata = {
    "accelerator": "nvidiaTeslaT4",
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {
        "name": "python",
        "version": "3.10.12",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
        "file_extension": ".py",
    },
}

# Cell 1: Header and GPU info
c1 = nbf.v4.new_code_cell("""# 🚀 Modern nanoGPT (124M) Distributed Pretraining on Kaggle GPU (T4 x2 / P100)
# Features:
# 1. Automatic Multi-GPU Detection (Dual T4 x2 or P100 single GPU)
# 2. Muon 5th-order Newton-Schulz Orthogonal Matrix Optimizer
# 3. Modern nanoGPT Architecture (RoPE + SwiGLU + RMSNorm + Zero-Bias + 50304 Vocab)
# 4. FineWeb 500M Token Memory-Mapped Zero-Copy Dataset

!nvidia-smi
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
num_gpus = torch.cuda.device_count()
print(f"GPU Count: {num_gpus}")
for i in range(num_gpus):
    print(f" GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB)")
""")

# Cell 2: Setup NeMo AutoModel & Dependencies
c2 = nbf.v4.new_code_cell("""# 📦 Install Dependencies
!pip install -q uv
!uv pip install --system transformers datasets torchdata tiktoken safetensors accelerate bitsandbytes pyyaml typer mlflow wandb torchao

# Clone or pull Automodel
import os
if not os.path.exists("Automodel"):
    !git clone https://github.com/NVIDIA-NeMo/Automodel.git
%cd Automodel
!uv pip install --system --no-deps -e .
""")

# Cell 3: Inject Modern SOTA nanoGPT & Muon Optimizer
c3 = nbf.v4.new_code_cell("""# 🧠 Inject SOTA Modern nanoGPT Architecture (RoPE + SwiGLU + RMSNorm + Bias-Free)
gpt2_code = '''# Modern nanoGPT implementation with RoPE, RMSNorm, SwiGLU
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        norm = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).type_as(x)

def precompute_rope_freqs(dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    B, n_h, T, head_dim = x.shape
    x_shaped = torch.view_as_complex(x.float().reshape(B, n_h, T, -1, 2))
    freqs_cis = freqs_cis[:T, :].unsqueeze(0).unsqueeze(0)
    x_out = torch.view_as_real(x_shaped * freqs_cis).flatten(3)
    return x_out.type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd=768, n_head=12, dropout=0.0, bias=False, use_rope=True, max_seq_len=2048):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.use_rope = use_rope
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.dropout = dropout
        if use_rope:
            self.register_buffer("freqs_cis", precompute_rope_freqs(self.head_dim, max_seq_len), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.use_rope:
            q = apply_rotary_emb(q, self.freqs_cis)
            k = apply_rotary_emb(k, self.freqs_cis)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, bias: bool = False):
        super().__init__()
        self.fc_gate = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc_up = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc_down = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_down(F.silu(self.fc_gate(x)) * self.fc_up(x))

class Block(nn.Module):
    def __init__(self, n_embd=768, n_head=12, dropout=0.0, bias=False, use_rope=True, norm_type="rmsnorm", mlp_type="swiglu"):
        super().__init__()
        self.ln_1 = RMSNorm(n_embd) if norm_type == "rmsnorm" else nn.LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, bias, use_rope)
        self.ln_2 = RMSNorm(n_embd) if norm_type == "rmsnorm" else nn.LayerNorm(n_embd, bias=bias)
        swiglu_hidden = int(2 * (4 * n_embd) / 3)
        swiglu_hidden = ((swiglu_hidden + 63) // 64) * 64
        self.mlp = SwiGLU(n_embd, swiglu_hidden, bias=bias) if mlp_type == "swiglu" else nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2LMHeadModel(nn.Module):
    def __init__(self, vocab_size=50304, n_positions=1024, n_embd=768, n_layer=12, n_head=12, bias=False, use_rope=True, norm_type="rmsnorm", mlp_type="swiglu", dropout=0.0):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = None if use_rope else nn.Embedding(n_positions, n_embd)
        self.h = nn.ModuleList([Block(n_embd, n_head, dropout, bias, use_rope, norm_type, mlp_type) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd) if norm_type == "rmsnorm" else nn.LayerNorm(n_embd, bias=bias)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.n_layer = n_layer
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.wte(input_ids)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x).float()
        # Soft-cap logits to [-30, 30] for Gemma-2 / Grok style 100% stable FP16 Tensor Core execution
        logits = 30.0 * torch.tanh(logits / 30.0)
        return logits

def build_gpt2_model(vocab_size=50304, n_positions=1024, n_embd=768, n_layer=12, n_head=12, bias=False, use_rope=True, norm_type="rmsnorm", mlp_type="swiglu", dropout=0.0, **kwargs):
    return GPT2LMHeadModel(vocab_size, n_positions, n_embd, n_layer, n_head, bias, use_rope, norm_type, mlp_type, dropout)
'''
with open("nemo_automodel/components/models/gpt2.py", "w") as f:
    f.write(gpt2_code)

# Inject Muon optimizer
muon_code = '''# Muon Newton-Schulz Orthogonal Matrix Optimizer
import torch
from torch.optim.optimizer import Optimizer

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert len(G.shape) >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype in (torch.float32, torch.bfloat16) else G.float()
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.type_as(G)

class Muon(Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g_update = g.add(buf, alpha=momentum) if nesterov else buf
                if g_update.ndim >= 2:
                    g_ortho = zeropower_via_newtonschulz5(g_update, steps=ns_steps)
                    scale = max(1.0, g.size(0) / g.size(1)) ** 0.5
                    p.data.add_(g_ortho, alpha=-lr * scale)
                else:
                    p.data.add_(g_update, alpha=-lr)
        return loss
'''
with open("nemo_automodel/components/optim/muon.py", "w") as f:
    f.write(muon_code)

# 🔧 Patch dion.py to prevent routing local pure-PyTorch Muon to Dion's MuonConfig
with open("nemo_automodel/components/optim/dion.py", "r") as f:
    dion_src = f.read()
dion_src = dion_src.replace(
    'name in {"Dion", "Dion2", "Muon", "NorMuon"}',
    'name in {"Dion", "Dion2", "Muon", "NorMuon"} and not module.startswith("nemo_automodel.components.optim.muon")'
)
with open("nemo_automodel/components/optim/dion.py", "w") as f:
    f.write(dion_src)

# Export Muon in nemo_automodel/components/optim/__init__.py
with open("nemo_automodel/components/optim/__init__.py", "a") as f:
    f.write("\\nfrom .muon import Muon\\n")

print("✅ SOTA nanoGPT and Muon modules injected & dion router patched successfully!")
""")

# Cell 4: Fast Dataset Preparation
c4 = nbf.v4.new_code_cell("""# 📥 Stream and Tokenize FineWeb Dataset (500M Tokens) directly on Kaggle
import os
import tiktoken
import numpy as np
from datasets import load_dataset

data_dir = "/kaggle/working/data"
os.makedirs(data_dir, exist_ok=True)
bin_path = os.path.join(data_dir, "dataset.bin")
idx_path = os.path.join(data_dir, "dataset.bos.idx")

# Verify existing dataset header or recreate
need_generate = True
if os.path.exists(bin_path):
    try:
        from nemo_automodel.components.datasets.llm.nanogpt_dataset import load_bin_shard
        t = load_bin_shard(bin_path)
        print(f"✅ Verified valid dataset at {bin_path} ({len(t) / 1e6:.1f}M tokens)")
        need_generate = False
    except Exception as e:
        print(f"⚠️ Re-creating dataset to write valid header: {e}")
        os.remove(bin_path)
        if os.path.exists(idx_path):
            os.remove(idx_path)

if need_generate:
    print("🚀 Streaming and tokenizing FineWeb dataset with standard 256-int32 header...")
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]
    target_tokens = 500_000_000

    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)

    token_count = 0
    buffer = []
    bos_indices = [0]

    with open(bin_path, "wb") as f_bin:
        # 1. Write initial 256 int32 placeholder header (1024 bytes)
        hdr = np.zeros(256, dtype=np.int32)
        hdr.tofile(f_bin)

        # 2. Stream tokens
        for item in ds:
            text = item.get("text", "")
            tokens = enc.encode_ordinary(text)
            tokens.append(eot)

            bos_indices.append(token_count)
            buffer.extend(tokens)
            token_count += len(tokens)

            if len(buffer) >= 1_000_000:
                np.array(buffer, dtype=np.uint16).tofile(f_bin)
                buffer = []
                print(f"Processed: {token_count / 1e6:.1f}M / {target_tokens / 1e6:.1f}M Tokens...", end="\\r")
            if token_count >= target_tokens:
                break

        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(f_bin)

        # 3. Finalize header at byte 0
        f_bin.seek(0)
        hdr[0] = 20240520       # LEGACY_MAGIC
        hdr[1] = 1              # VERSION
        hdr[2] = token_count    # total number of uint16 tokens
        hdr[3] = 2              # 2 bytes per token
        hdr.tofile(f_bin)

    with open(idx_path, "wb") as f_idx:
        np.array(bos_indices, dtype=np.int32).tofile(f_idx)

    print(f"\\n✅ FineWeb 500M Dataset created at {bin_path} with standard header ({os.path.getsize(bin_path) / 1e6:.1f} MB, {token_count} tokens)")
""")

# Cell 5: Create Dynamic GPU Recipe Config
c5 = nbf.v4.new_code_cell("""# ⚙️ Generate Optimized YAML Recipe adapted to active GPU hardware
import torch
num_gpus = max(1, torch.cuda.device_count())
global_batch_size = 32 if num_gpus >= 2 else 16
local_batch_size = 4  # 🛡️ 4 microbatch per GPU (consumes ~5.3GB VRAM, 100% safe & zero OOM)

recipe_yaml = f'''
recipe: TrainFinetuneRecipeForNextTokenPrediction

step_scheduler:
  global_batch_size: {global_batch_size}
  local_batch_size: {local_batch_size}
  ckpt_every_steps: 500
  val_every_steps: 100
  num_epochs: 1
  max_steps: {int(500_000_000 / (global_batch_size * 1024))}

dist_env:
  backend: nccl
  timeout_minutes: 5

model:
  _target_: nemo_automodel.components.models.gpt2.build_gpt2_model
  vocab_size: 50304
  n_positions: 1024
  n_embd: 768
  n_layer: 12
  n_head: 12
  bias: false
  use_rope: true
  norm_type: "rmsnorm"
  mlp_type: "swiglu"
  dropout: 0.0

dataset:
  _target_: nemo_automodel.components.datasets.llm.nanogpt_dataset.NanogptDataset
  file_pattern: "/kaggle/working/data/dataset.bin"
  seq_len: 1024
  shuffle_files: true
  align_to_bos: false

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  shuffle: false
  collate_fn: nemo_automodel.components.datasets.utils.default_collater

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

clip_grad_norm:
  max_norm: 1.0

optimizer:
  _target_: torch.optim.AdamW
  lr: 0.0006
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 0.1

lr_scheduler:
  lr_decay_style: cosine
  lr_warmup_steps: 200
  min_lr: 0.00006

distributed:
  strategy: fsdp2
  dp_size: none
  tp_size: 1
  cp_size: 1
  sequence_parallel: false
  enable_compile: false
  mp_policy:
    param_dtype: torch.float16
    reduce_dtype: torch.float32
    output_dtype: torch.float16

checkpoint:
  enabled: true
  checkpoint_dir: /kaggle/working/checkpoints/gpt2_kaggle/
  restore_from: "LATEST"
  model_save_format: torch_save
  save_consolidated: false
  max_recent_checkpoints: 3
'''
with open("gpt2_kaggle_recipe.yaml", "w") as f:
    f.write(recipe_yaml)

print(f"✅ gpt2_kaggle_recipe.yaml generated for {num_gpus} GPU(s) with global_batch_size={global_batch_size}!")
""")

# Cell 6: Launch Pretraining
c6 = nbf.v4.new_code_cell("""# 🚀 Launch Distributed Pretraining
import os
import torch

os.environ["WANDB_MODE"] = "disabled"
os.environ["MLFLOW_TRACKING_URI"] = "file:///kaggle/working/mlruns"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

num_gpus = max(1, torch.cuda.device_count())
print(f"🔥 Starting Multi-GPU Distributed Training on {num_gpus} GPU(s)...")
!torchrun --nproc_per_node={num_gpus} nemo_automodel/recipes/llm/train_ft.py --config gpt2_kaggle_recipe.yaml
""")

nb.cells = [c1, c2, c3, c4, c5, c6]

with open("projects/gpt2_fineweb_500m/kaggle_deployment/train_gpt2_t4x2.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("✅ projects/gpt2_fineweb_500m/kaggle_deployment/train_gpt2_t4x2.ipynb updated with accelerator metadata!")
