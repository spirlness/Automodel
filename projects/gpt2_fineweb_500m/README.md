# GPT-2 FineWeb 500M 训练与性能指南

本目录是一个基于 NeMo AutoModel 的 GPT-2 风格语言模型预训练实验。默认配方使用
FineWeb 5 亿 token 二进制数据、RoPE、RMSNorm、SwiGLU、权重共享词表头和 Muon
优化器。本文档以
[`config/gpt2_fineweb_500m.yaml`](config/gpt2_fineweb_500m.yaml) 为当前唯一默认配置。

## 当前已验证结果

实测环境为 RTX 3060 Laptop GPU（6 GiB）、PyTorch 2.10 CUDA、单卡 BF16。完成首次
`torch.compile` 与 CUDA/Triton 预热后，默认配方的真实 FineWeb 训练结果为：

| 指标 | 实测值 | 说明 |
| --- | ---: | --- |
| 稳定吞吐 | 约 21,000 tokens/s | 不计第一步编译和数据页缓存预热 |
| MFU | 约 26% | 固定 FLOPs/token 与 BF16 Tensor Core 峰值估算 |
| 峰值显存 | 约 3.83 GiB | `local_batch_size=8`，两次梯度累积 |
| 全局 batch | 16 sequences | 每步处理 `16 × 1024 = 16,384` token |
| 上下文长度 | 1,024 | 数据、模型和 MFU 计算使用同一长度 |
| 可训练参数 | 123,587,328 | 约 124M 参数 |

12 步真实回归中，loss 从约 `10.997` 降至约 `8.258`，梯度范数全程有限。这证明默认
路径能正常完成数据读取、前向、反向、梯度裁剪和优化器更新；它不是 5 亿 token 完整收敛的
质量结论。

> 当前硬件下，在保留完整 5 次 Newton–Schulz Muon 正交化和真实 FineWeb 训练的条件下，
> 稳定 MFU 尚未达到 30%。不要为提高数字而直接降低 `ns_steps`。

## 目录结构

```text
projects/gpt2_fineweb_500m/
├── README.md                         # 本文档
├── config/gpt2_fineweb_500m.yaml     # 默认训练配方
├── data/fineweb_500M_max_tokens_500M/dataset.bin
├── tools/nanogpt_data_processor.py   # FineWeb 预处理
├── checkpoints/                      # 本地与 Kaggle checkpoint 产物
├── kaggle_deployment/                # Kaggle notebook 资源
└── scripts/                          # 运维和下载脚本
```

核心实现位于框架主包，项目目录不复制模型代码：

| 组件 | 源码位置 | 职责 |
| --- | --- | --- |
| GPT-2 模型 | `nemo_automodel/components/models/gpt2.py` | RoPE、RMSNorm、SwiGLU、SDPA、共享词表头 |
| Muon | `nemo_automodel/components/optim/muon.py` | transformer 矩阵的 Newton–Schulz 正交化更新 |
| 参数分组 | `nemo_automodel/components/optim/optimizer.py` | Muon 与 AdamW 的安全分组 |
| 融合损失 | `nemo_automodel/components/loss/linear_ce.py` | 融合词表投影、softmax 与交叉熵 |
| 训练循环 | `nemo_automodel/recipes/llm/train_ft.py` | 数据、反向、裁剪、日志、checkpoint |

## 环境准备

从仓库根目录运行。默认配方启用 `cut-cross-entropy`，该包在 `dev` 依赖组中：

```bash
uv sync --locked --group dev --extra fa --inexact
```

确认 GPU 可见：

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

第一个输出必须为 `True`。本模型使用 PyTorch SDPA，安装 `flash-attn` 不会令模型自动
切换为 FlashAttention-2。

## 准备 FineWeb 数据

默认数据文件为：

```text
projects/gpt2_fineweb_500m/data/fineweb_500M_max_tokens_500M/dataset.bin
```

未生成数据时，执行：

```bash
uv run python projects/gpt2_fineweb_500m/tools/nanogpt_data_processor.py \
  --dataset HuggingFaceFW/fineweb \
  --set-name sample-10BT \
  --max-tokens 500M
```

预处理器将原始文本 token 化成内存映射二进制数据。配置中的 `max_steps=30517` 等于：

```text
floor(500,000,000 / (global_batch_size 16 × sequence_length 1024))
```

最后不能组成完整全局 batch 的少量 token 不会参与这一 epoch。

## 启动训练

当前基准面向单张 GPU，使用一个进程：

```bash
uv run automodel projects/gpt2_fineweb_500m/config/gpt2_fineweb_500m.yaml \
  --nproc-per-node 1
```

默认行为：

- 每 500 步保存 checkpoint；
- 每 100 步执行验证；
- checkpoint 目录为 `checkpoints/gpt2_fineweb_500m/`；
- 日志记录 loss、grad norm、学习率、显存和 tokens/s。

### 12 步真实训练回归

修改模型、损失、优化器或性能设置后，使用以下命令进行最小真实训练验证。它禁用 checkpoint
和验证，避免干扰吞吐：

```bash
uv run automodel projects/gpt2_fineweb_500m/config/gpt2_fineweb_500m.yaml \
  --nproc-per-node 1 \
  --step_scheduler.max_steps=12 \
  --step_scheduler.ckpt_every_steps=1000000 \
  --step_scheduler.val_every_steps=1000000 \
  --step_scheduler.save_checkpoint_every_epoch=false \
  --checkpoint.enabled=false \
  --lr_scheduler.lr_warmup_steps=2
```

该命令故意缩短 warmup，便于在 12 步内检查学习率与数值。正式训练仍使用 YAML 中的
`lr_warmup_steps: 200`。

## 配方细节

### 模型

| 配置 | 值 | 用途 |
| --- | ---: | --- |
| `vocab_size` | 50,304 | 64 的倍数，适合 Tensor Core 矩阵形状 |
| `n_positions` | 1,024 | 最大训练上下文；超长输入会显式报错 |
| `n_embd/n_layer/n_head` | 768 / 12 / 12 | 约 124M GPT-2 规模 |
| `use_rope` | `true` | rotary position embedding |
| `norm_type` | `rmsnorm` | 较低的归一化开销 |
| `mlp_type` | `swiglu` | 融合 gate/up 投影的 SwiGLU FFN |
| `bias/dropout` | `false/0.0` | 预训练吞吐配置 |
| `attn_implementation` | `sdpa` | PyTorch scaled-dot-product attention |
| `torch_dtype` | `bfloat16` | 单卡保存 BF16 参数，避免每层重复权重 autocast |

`wte` 与 `lm_head` 权重共享。普通前向返回 `[batch, sequence, vocab]` logits；融合损失
通过 `logits_to_keep=1` 请求最终 `[batch, sequence, hidden]` hidden states，避免训练时
物化完整词表 logits。

### 批大小与精度

```yaml
step_scheduler:
  global_batch_size: 16
  local_batch_size: 8

performance:
  float32_matmul_precision: high
```

单卡下，梯度累积为 `16 / 8 = 2` 次。`local_batch_size=8` 是本机的可靠甜点点：比 2 或
4 更快，且显存仍有余量。`float32_matmul_precision: high` 启用 Ampere 上适用的 TF32
矩阵计算；BF16 路径仍是本配方主要计算精度。

`distributed.enable_compile: true` 会对 transformer 层应用 `torch.compile`。首次训练步
包含编译开销，不能用于吞吐或 MFU 对比。

### 融合交叉熵

```yaml
loss_fn:
  _target_: nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy
  logit_softcapping: 30.0
```

融合损失将 LM head 投影、softmax 与交叉熵合并，避免构造
`[batch, sequence, 50304]` 全量 logits。`logit_softcapping: 30.0` 保持原模型的 soft-cap
目标。其输入会对齐为 BF16，同时梯度仍回传至原始权重。

### Muon 的安全参数分组

| 参数类别 | 优化器 | 学习率 |
| --- | --- | ---: |
| transformer 线性矩阵 | Muon | `0.02` |
| token embedding / 共享 lm head | AdamW | `0.0006` |
| RMSNorm 等标量或向量参数 | AdamW | `0.0006` |

embedding 不会进入 Newton–Schulz 正交化。默认 `ns_steps=5` 保留完整 Muon 更新；短测中
降低该值确实更快，但会改变优化器数学与潜在收敛质量。

BF16 梯度裁剪会缩放真实梯度，而不是仅缩放 FP32 临时副本；`clip_grad_norm.max_norm: 1.0`
在反向后、优化器更新前生效。

## MFU 口径与性能结果

本模型的每 token 估算训练 FLOPs 为：

```text
linear FLOPs/token    = 72 × layers × hidden² + 6 × hidden × vocab
attention FLOPs/token = 6 × layers × hidden × sequence
total                 = 798,031,872 FLOPs/token
```

代入 `layers=12`、`hidden=768`、`vocab=50,304`、`sequence=1,024`，MFU 使用：

```text
MFU = tokens_per_second × 798,031,872 / GPU_BF16_dense_peak_FLOPs
```

约 21k tokens/s 对应本机约 26% MFU。这个估算只用于相同模型形状、序列长度、全局 batch
和 GPU 峰值口径下的横向比较。

| 已完成优化 | 影响 |
| --- | --- |
| 默认 TF32 | 消除 TF32 未启用告警，优化残留 FP32 GEMM |
| 融合 SwiGLU gate/up | 减少 MLP 的独立线性调用与中间张量 |
| 融合线性交叉熵 | 不物化全词表训练 logits，降低显存 |
| BF16 参数存储 | 避免单卡 FSDP 跳过时反复转换 FP32 权重 |
| `local_batch_size=8` | 从 8 次累积减少至 2 次，保持 global batch=16 |
| BF16 梯度裁剪修复 | 恢复裁剪正确性并移除无效的缩放副本 |

早期稳定吞吐约 8.5k tokens/s、MFU 约 10.6%；当前默认结果约 21k tokens/s、MFU 约 26%。

## 已评估但未采用的方案

| 方案 | 实测结果 | 未采用原因 |
| --- | --- | --- |
| `local_batch_size=16` | 显存约 5.9 GiB，出现严重首步/后续退化 | 接近 6 GiB 容量，无法稳定使用 |
| 本地 Muon `ns_steps=3` | 约 21.9k tokens/s，MFU 约 27.1% | 修改正交化精度，仍未达 30% |
| 本地 Muon `ns_steps=1` | 约 22.9k tokens/s，MFU 约 28.3% | 牺牲优化质量，不适合作为默认 |
| Dion `use_triton=true` | 两次停在超长 CPU/Triton 编译 | 当前环境未能在合理时间进入可测训练步 |
| `torch.compile(optimizer.step)` | 预热超过一分钟且显存接近上限 | 不适合 6 GiB 日常训练 |
| FA2 包 | 本模型训练端与 SDPA 无明显优势 | 本地实现明确使用 SDPA |

因此默认配置优先保持真实训练稳定性、完整 Muon 更新与单卡可复现性，而非仅提高 MFU 数字。

## 常见问题

### 缺少 `cut-cross-entropy`

执行：

```bash
uv sync --locked --group dev --extra fa --inexact
```

临时排查时可改用普通交叉熵，但显存和吞吐会变差：

```bash
uv run automodel projects/gpt2_fineweb_500m/config/gpt2_fineweb_500m.yaml \
  --nproc-per-node 1 \
  --loss_fn._target_=nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy
```

### TE/Apex、grouped_gemm 或 torchao 警告

它们是可选组件提示，不阻止当前 SDPA + BF16 + 本地 Muon 路径。RTX 3060 单卡默认训练
不依赖 Transformer Engine、Apex 或 grouped_gemm。

### FA2 已安装却未被调用

这是预期行为：配方设定 `attn_implementation: sdpa`，本地 GPT-2 没有 `flash-attn` 的单独
后端分支。

### WSL 上多 worker 数据加载失败

部分挂载文件系统不支持 `StatefulDataLoader` 多 worker 所需的共享资源。默认配置不强制多
worker；保持默认单进程加载即可。

### checkpoint 恢复失败

早期单组 Muon checkpoint 的 optimizer state 与当前分组优化器不兼容。仅对当前分组配方
生成的 checkpoint 设置 `checkpoint.restore_from`；其他 checkpoint 应启动新训练。

## Kaggle checkpoint

```bash
uv run python projects/gpt2_fineweb_500m/scripts/download_kaggle_checkpoint.py
```

默认输出为 `projects/gpt2_fineweb_500m/checkpoints/gpt2_kaggle/`。checkpoint、原始数据和
源码应保持分离。

## 修改后的验证清单

1. 模型接口回归：

   对应测试位于
   `tests/unit_tests/models/test_local_gpt2.py`，覆盖 BF16 前向、融合损失 hidden-state
   输出、RoPE 上下文长度边界、显式 SDPA 限制、BF16 模型构建和最小 CUDA 更新。
   当前桌面环境的 pytest 分片插件被预设为一个空分片，直接运行 pytest 会显示
   `Running 0 items in this shard`，不代表测试通过。请在未强制分片的 CI 或修复该本地
   pytest 配置后执行此测试文件。

2. 格式与静态检查：

   ```bash
   uv run ruff format --check nemo_automodel/components/models/gpt2.py \
     nemo_automodel/components/optim/muon.py \
     nemo_automodel/components/loss/linear_ce.py
   uv run ruff check nemo_automodel/components/models/gpt2.py \
     nemo_automodel/components/optim/muon.py \
     nemo_automodel/components/loss/linear_ce.py
   ```

3. 运行上文 12 步真实 FineWeb 回归，确认预热后吞吐接近基线、loss/grad norm 为有限值、显存
   不超容量，并且没有残留训练进程。

不要将第一步编译时间、不同上下文长度、不同 global batch，或降低 Muon `ns_steps` 的结果，
与本文约 26% MFU 基线直接比较。
