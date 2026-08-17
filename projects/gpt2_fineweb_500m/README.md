# GPT-2 FineWeb 500M 预训练项目

本项目基于 **NeMo AutoModel** 框架构建，实现现代增强版 **GPT-2（约 124M 参数）** 在 **FineWeb 500M tokens** 上的自监督预训练。

项目专注于轻量、高效、极简依赖的基准训练管线：
- **现代 Transformer 架构**：内置 RoPE 旋转位置编码、RMSNorm、SwiGLU 门控前馈网络、Zero-Bias（无偏置纯 GEMM）、50,304 硬件对齐词表；
- **原生 PyTorch 优化**：使用 PyTorch 内置 SDPA（Scaled Dot-Product Attention）和 TF32 / bfloat16 混合精度；
- **标准训练组件**：基于 `TrainFinetuneRecipeForNextTokenPrediction`、`MaskedCrossEntropy`、`torch.optim.AdamW` 和 DDP 数据并行；
- **流式二进制数据处理**：单进程顺序流水线，生成带 BOS 索引的内存映射二进制数据分块（`.bin` + `.bos.idx`）；
- **极简工程依赖**：不依赖 Transformer Engine、Apex、grouped_gemm、flash-attn 独立扩展、torchao C++ 扩展或 `torch.compile`。

---

## 1. 文件结构

```text
projects/gpt2_fineweb_500m/
├── README.md                          # 项目说明与实战文档
├── config/
│   └── gpt2_fineweb_500m.yaml         # 训练超参数与组件配置
├── data/                              # FineWeb 二进制数据集输出目录
│   └── fineweb_500M_max_tokens_500M/  # 包含 dataset.bin 与 dataset.bos.idx
├── checkpoints/                       # 训练检查点与 JSONL 训练日志
├── tools/
│   └── nanogpt_data_processor.py      # FineWeb 单进程流式预处理工具
└── scripts/                           # 项目快捷脚本
    ├── prepare_data.sh                # 一键数据下载与分词脚本
    ├── smoke_test.sh                  # 2 步快速冒烟测试脚本
    ├── train.sh                       # 单卡 / 多卡训练启动脚本
    └── eval_mfu.sh                    # 100 步 MFU 性能与吞吐评测脚本
```

---

## 2. 环境准备

从仓库根目录执行安装并同步虚拟环境：

```bash
uv sync --locked --no-default-groups --inexact
```

验证 PyTorch 与 GPU 可用性：

```bash
uv run python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 3. 数据准备

### 3.1 在线流式下载与分块转换
使用 `nanogpt_data_processor.py` 从 Hugging Face 流式拉取 FineWeb 数据集，使用 GPT-2 分词器编码，并输出为 NanoGPT 二进制格式：

```bash
# 方式 1：使用便捷脚本
bash projects/gpt2_fineweb_500m/scripts/prepare_data.sh HuggingFaceFW/fineweb sample-10BT 500M

# 方式 2：使用完整 Python 命令
uv run python projects/gpt2_fineweb_500m/tools/nanogpt_data_processor.py \
  --dataset HuggingFaceFW/fineweb \
  --set-name sample-10BT \
  --output-dir projects/gpt2_fineweb_500m/data/fineweb_500M \
  --max-tokens 500M \
  --max-length 1024
```

### 3.2 离线 Parquet 转换
如果已在本地下载了 Parquet 文件，可通过 `--local-parquet-dir` 离线快速处理：

```bash
uv run python projects/gpt2_fineweb_500m/tools/nanogpt_data_processor.py \
  --local-parquet-dir /path/to/parquet/files \
  --output-dir projects/gpt2_fineweb_500m/data/fineweb_500M \
  --max-tokens 500M \
  --max-length 1024
```

### 3.3 数据产物说明
- 预处理器采用严格单进程顺序处理，内存占用极低且无多进程队列死锁隐患；
- `--max-tokens` 为严格全局 token 上限，`--max-length` 为单文档截断长度（首位自动插入 BOS token `50256`）；
- 输出文件：
  - `dataset.bin`：`uint16` 格式的连续 token 序列，包含 256 个 int32 的 header（Magic: `278895051`, Version: `1`）；
  - `dataset.bos.idx`：每个文档起始 BOS token 的绝对索引位置；
  - `args.json`：生成时的参数元数据。

---

## 4. 训练超参数详解

训练配置文件位于 [`config/gpt2_fineweb_500m.yaml`](file:///home/lee/Automodel/projects/gpt2_fineweb_500m/config/gpt2_fineweb_500m.yaml)：

| 配置模块 | 关键参数 | 设定值 | 说明 |
| :--- | :--- | :--- | :--- |
| **模型架构** | `vocab_size` | `50304` | 硬件 Tensor Core 64 倍数对齐 |
| | `n_positions` / `seq_len` | `1024` | 训练序列长度 |
| | `n_embd` / `n_layer` / `n_head` | `768` / `12` / `12` | 标准 124M 规格（Head Dim = 64） |
| | `norm_type` / `mlp_type` | `"rmsnorm"` / `"swiglu"` | 现代 LLM 结构升级 |
| | `bias` / `use_rope` | `false` / `true` | 无偏置纯矩阵乘法 + 旋转位置编码 |
| | `torch_dtype` | `bfloat16` | 混合精度计算 |
| **批次与步数** | `global_batch_size` | `16` | 全局 batch 大小（每步 16,384 tokens） |
| | `local_batch_size` | `8` | 单卡 micro-batch 大小（单卡累计 2 次梯度） |
| | `max_steps` | `30517` | $\lfloor 500,000,000 / (16 \times 1024) \rfloor$ |
| **优化器** | `optimizer` | `torch.optim.AdamW` | 学习率 `6e-4`, $\beta = (0.9, 0.95)$, weight_decay `0.1` |
| | `clip_grad_norm` | `1.0` | 梯度裁剪范数 |
| **调度器** | `lr_scheduler` | `cosine` | 200 步线性 Warmup，最小学习率 `6e-5`（峰值 10%） |
| **并行与检查点** | `distributed.strategy` | `ddp` | 标准分布式数据并行 |
| | `ckpt_every_steps` | `500` | 每 500 步保存一次检查点，保留最近 3 份 |

---

## 5. 启动训练

### 5.1 单卡快速冒烟测试（2 步验证）
在启动长时间训练前，建议先执行 2 步短测，验证数据加载、模型前反向传播与 AdamW 权重更新是否正常：

```bash
# 方式 1：直接运行便捷脚本
bash projects/gpt2_fineweb_500m/scripts/smoke_test.sh

# 方式 2：使用 automodel CLI 命令
uv run automodel projects/gpt2_fineweb_500m/config/gpt2_fineweb_500m.yaml \
  --nproc-per-node 1 \
  --dataset.file_pattern=projects/gpt2_fineweb_500m/data/fineweb_500M_max_tokens_500M/dataset.bin \
  --step_scheduler.max_steps=2 \
  --checkpoint.enabled=false
```

### 5.2 性能与 MFU 基准评测（100 步测试）
```bash
# 运行 100 步评测并自动计算吞吐与 MFU
bash projects/gpt2_fineweb_500m/scripts/eval_mfu.sh 100
```

### 5.3 单卡全量训练
```bash
# 便捷脚本
bash projects/gpt2_fineweb_500m/scripts/train.sh 1

# 或直接使用 CLI
uv run automodel projects/gpt2_fineweb_500m/config/gpt2_fineweb_500m.yaml \
  --nproc-per-node 1
```

### 5.4 多卡分布式训练（例如 2 卡 DDP）
```bash
# 便捷脚本（指定卡数）
bash projects/gpt2_fineweb_500m/scripts/train.sh 2

# 或直接使用 CLI
uv run automodel projects/gpt2_fineweb_500m/config/gpt2_fineweb_500m.yaml \
  --nproc-per-node 2
```
*注：当卡数增加时，确保 `global_batch_size`（16）能被 `local_batch_size × dp_size` 整除。若使用 4 卡，可将 `local_batch_size` 设为 `4`。*

---

## 6. 检查点管理与断点续训

### 6.1 检查点保存路径
训练检查点默认保存至：
```text
projects/gpt2_fineweb_500m/checkpoints/gpt2_fineweb_500m/
├── LATEST                           # 指向最新检查点目录的文本指针
├── epoch_0_step_500/
├── epoch_0_step_1000/
└── training.jsonl                   # 训练过程详细指标日志
```

### 6.2 断点续训 (Resume Training)
若训练意外中断，可以通过指定 `--checkpoint.restore_from` 从最近检查点无缝恢复：

```bash
uv run automodel projects/gpt2_fineweb_500m/config/gpt2_fineweb_500m.yaml \
  --nproc-per-node 1 \
  --checkpoint.restore_from=projects/gpt2_fineweb_500m/checkpoints/gpt2_fineweb_500m/epoch_0_step_1000
```
*(注意：请勿尝试从旧版 Muon / FSDP2 格式的旧检查点中恢复优化器状态)*

---

## 7. 性能与算力利用率 (MFU) 说明

- **单步 FLOPs 理论值**：
  $$\text{FLOPs}_{\text{step}} = 16384 \times [12 \times (72 \times 768^2 + 6 \times 1024 \times 768) + 6 \times 768 \times 50304] \approx \mathbf{13.075\text{ TFLOPs}}$$
- **MFU 计算公式**：
  $$\text{MFU} = \frac{\text{Step FLOPs}}{\text{GPUs} \times \text{Step Time (s)} \times \text{Peak TFLOPs}} \times 100\%$$
- **训练日志指标**：
  `training.jsonl` 中记录了每步的 `loss`、`grad_norm`、`lr`、`mem`（显存占用 GB）和 `tps`（每秒处理 tokens 数）。

---

## 8. 常见排查指南

1. **AssertionError: `assert self.max_lr >= self.min_lr`**：
   确保 `lr_scheduler.min_lr`（当前为 `6e-5`）小于等于 `optimizer.lr`（当前为 `6e-4`）。
2. **数据路径找不到 (FileNotFoundError)**：
   检查 `dataset.file_pattern` 是否指向正确的 `projects/gpt2_fineweb_500m/data/fineweb_500M_max_tokens_500M/dataset.bin`。
3. **显存溢出 (OOM)**：
   若显存较小（如 6GB/8GB 显卡），可将 `step_scheduler.local_batch_size` 调小至 `4` 或 `2`，梯度累积步数会自动增加以保持 `global_batch_size: 16` 不变。
