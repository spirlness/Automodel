# GPT-2 FineWeb：最小 AutoModel 训练路径

这个项目用于在 FineWeb 上训练约 124M 参数的 GPT-2 风格模型。当前默认优先保证
Kaggle 可执行性和可排错性，只使用基础组件：

- 模型：项目内的 `build_gpt2_model`，PyTorch SDPA 注意力；
- 损失：`MaskedCrossEntropy`，物化普通 logits；
- 优化器：`torch.optim.AdamW`；
- 多卡：PyTorch DDP；
- 精度：T4 使用 FP16，保留 TF32 开关；
- 数据：单进程顺序流式读取和写入。

本路径不依赖 Transformer Engine、Apex、grouped_gemm、flash-attn、torchao C++ 扩展、
本地 Muon、FSDP2 或 `torch.compile`。这些功能可以在框架其他配方中使用，但不属于本项目
的 Kaggle 最小验证面。

## 文件结构

```text
projects/gpt2_fineweb_500m/
├── config/gpt2_fineweb_500m.yaml       # 本地默认配方
├── config/gpt2_fineweb_t4x2.yaml       # Kaggle 双 T4 配方
├── tools/nanogpt_data_processor.py     # 顺序 FineWeb 预处理器
└── kaggle_deployment/
    ├── generate_nb.py                  # Notebook 生成器
    ├── train_gpt2_t4x2.ipynb           # 提交到 Kaggle 的 Notebook
    └── kernel-metadata.json
```

## 本地验证

从仓库根目录安装基础依赖：

```bash
uv sync --locked --no-default-groups --inexact
```

生成一个小数据集：

```bash
uv run python projects/gpt2_fineweb_500m/tools/nanogpt_data_processor.py \
  --dataset HuggingFaceFW/fineweb \
  --set-name sample-10BT \
  --output-dir /tmp/fineweb_smoke \
  --max-tokens 1M \
  --max-length 1024
```

运行单卡短测：

```bash
uv run automodel projects/gpt2_fineweb_500m/config/gpt2_fineweb_500m.yaml \
  --nproc-per-node 1 \
  --dataset.file_pattern=/tmp/fineweb_smoke_max_tokens_1M/dataset.bin \
  --step_scheduler.max_steps=2 \
  --checkpoint.enabled=false
```

预处理器的 `--max-tokens` 是整个数据集的严格上限，`--max-length` 是单文档上限并包含
插入的 BOS token。最终 `.bin` 文件的 header 会记录实际写入 token 数。

## Kaggle 双 T4

`config/gpt2_fineweb_t4x2.yaml` 使用：

```yaml
distributed:
  strategy: ddp

optimizer:
  _target_: torch.optim.AdamW
```

每张卡的 `local_batch_size=4`，全局 batch 为 8。训练 10 亿 token 时：

```text
floor(1,000,000,000 / (8 × 1,024)) = 122,070 steps
```

Notebook 的执行顺序是固定的：

1. 检查是否真的分配了两张 T4；
2. clone 并 checkout 固定 commit SHA；
3. `uv sync` 安装基础环境；
4. 处理 1M token smoke 数据；
5. 从 Kaggle YAML 构造模型、损失、优化器和 DDP 配置；
6. 用两进程完成一个真实前向、反向和 AdamW 更新；
7. smoke 成功后才处理 1B token 并开始正式训练。

生成 Notebook：

```bash
uv run python projects/gpt2_fineweb_500m/kaggle_deployment/generate_nb.py
```

将 `projects/gpt2_fineweb_500m/kaggle_deployment/` 作为 Kaggle Kernel 目录提交。正式训练
只保留最近 3 个 checkpoint，目录默认为 `/kaggle/working/checkpoints`。Kaggle 需要开启
Internet；FineWeb 是公开数据集，不设置 `HF_TOKEN` 也可以运行。如果设置了环境变量，
预处理器会自动使用它，但不会打印 token 内容。

## 出错时的排查顺序

先看 smoke-preprocess 是否出现以下日志：

```text
Opening HuggingFaceFW/fineweb/sample-10BT:train with streaming=True
Writing .../dataset.bin
Dataset created: ... (1000000 tokens ...)
```

然后看 smoke-config 是否输出 `strategy=ddp`，最后确认 smoke-train 的两个 rank 都完成
第一步。若预处理失败，优先检查 Kaggle Internet、磁盘空间和 Hugging Face Hub 访问，
不要先安装 TE/Apex/FA2；它们不参与这条路径。

现有 Muon/FSDP2 配方创建的 optimizer checkpoint 与当前 AdamW 配方不兼容。切换到本配置
后应从新运行开始，不要恢复旧 optimizer state。
