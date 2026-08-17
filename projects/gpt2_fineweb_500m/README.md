# GPT-2 FineWeb 500M

这是一个基于 NeMo AutoModel 的 GPT-2 风格语言模型预训练实验，默认使用约 124M 参数、
FineWeb 500M tokens 和基础训练组件：

- 项目内的 `build_gpt2_model`；
- PyTorch SDPA 注意力；
- `MaskedCrossEntropy`；
- `torch.optim.AdamW`；
- DDP 数据并行；
- 顺序流式 FineWeb 预处理。

训练路径不依赖 Transformer Engine、Apex、grouped_gemm、flash-attn、torchao C++ 扩展、
本地 Muon、FSDP2 或 `torch.compile`。

## 文件结构

```text
projects/gpt2_fineweb_500m/
├── README.md
├── config/gpt2_fineweb_500m.yaml
├── data/                              # FineWeb 二进制数据
├── checkpoints/                       # 训练 checkpoint
├── tools/nanogpt_data_processor.py    # FineWeb 预处理器
└── scripts/                           # 项目运维脚本
```

## 安装

从仓库根目录运行：

```bash
uv sync --locked --no-default-groups --inexact
```

确认 GPU：

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 准备数据

生成 500M tokens：

```bash
uv run python projects/gpt2_fineweb_500m/tools/nanogpt_data_processor.py \
  --dataset HuggingFaceFW/fineweb \
  --set-name sample-10BT \
  --output-dir projects/gpt2_fineweb_500m/data/fineweb_500M \
  --max-tokens 500M \
  --max-length 1024
```

预处理器是单进程顺序流水线，不使用 worker、队列或 Future，因此内存和生命周期更容易
排查。`--max-tokens` 是整个数据集的严格上限；`--max-length` 是单文档上限，包含插入的
BOS token。`.bin` 文件 header 会记录实际写入 token 数。

## 启动训练

单卡短测：

```bash
uv run automodel projects/gpt2_fineweb_500m/config/gpt2_fineweb_500m.yaml \
  --nproc-per-node 1 \
  --dataset.file_pattern=projects/gpt2_fineweb_500m/data/fineweb_max_tokens_500M/dataset.bin \
  --step_scheduler.max_steps=2 \
  --checkpoint.enabled=false
```

正式训练：

```bash
uv run automodel projects/gpt2_fineweb_500m/config/gpt2_fineweb_500m.yaml \
  --nproc-per-node 1
```

默认配置使用 global batch 16、sequence length 1024，并按完整 global batch 计算训练步数。
checkpoint 保存在 `projects/gpt2_fineweb_500m/checkpoints/gpt2_fineweb_500m/`，最多保留最近
3 个 checkpoint。

## 排查顺序

1. 先用小 token 预算确认 tokenizer 和 `.bin` writer 可用。
2. 检查 `.bin` header 的 token 数、magic 和 dtype 字段。
3. 用 `--step_scheduler.max_steps=2` 验证模型前向、反向和 AdamW 更新。
4. 确认短测通过后再启动完整数据和长时间训练。

如果训练从旧 Muon/FSDP2 配方切换而来，不要恢复旧 optimizer state；应从新运行开始。
