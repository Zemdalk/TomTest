# TomTest

基于 vLLM serve 的 Theory-of-Mind 基准评测框架，支持多数据集、多 prompt 模板的批量并行评测。

## 数据集下载

数据集托管于 [TomTraining/TomDatasets](https://huggingface.co/datasets/TomTraining/TomDatasets)，下载到本地 `TomDatasets/` 目录：

```bash
# 方法一：huggingface_hub（推荐）
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="TomTraining/TomDatasets",
    repo_type="dataset",
    local_dir="TomDatasets",
)
EOF
```

```bash
# 方法二：git lfs（需要安装 git-lfs）
git lfs install
git clone https://huggingface.co/datasets/TomTraining/TomDatasets TomDatasets
```

下载完成后目录结构如下：

```
TomDatasets/
├── Tomato/
│   ├── train/
│   └── test/
├── ToMBench/
│   └── test/
├── ToMQA/
│   ├── train/
│   ├── validation/
│   └── test/
└── ...
```

## 安装依赖

```bash
pip install -r requirements.txt
pip install datasets vllm
```

## 快速开始

### 1. 启动 vLLM serve

```bash
vllm serve /path/to/your/model \
    --port 8000 \
    --tensor-parallel-size 1 \ # 1 for single GPU, 2 for two GPUs
    --gpu-memory-utilization 0.8
```

### 2. 运行评测

**最小命令（全量 test split）：**

```bash
python run.py \
    --model my-model \
    --api-url http://localhost:8000/v1
```

**快速冒烟（每个 split 取 50 条）：**

```bash
python run.py \
    --model my-model \
    --api-url http://localhost:8000/v1 \
    --eval-phase screen
```

**只跑指定数据集 + 指定 prompt 模板：**

```bash
python run.py \
    --model my-model \
    --api-url http://localhost:8000/v1 \
    --dataset-filter Tomato ToMBench \
    --prompt-names "Standard Zero-shot" "BDI-ToM Agent"
```

**保存每条预测结果：**

```bash
python run.py \
    --model my-model \
    --api-url http://localhost:8000/v1 \
    --predictions-jsonl result/preds.jsonl
```

### 3. 查看结果

| 文件 | 内容 |
|---|---|
| `result/results_table.md` | 汇总准确率表格（Markdown） |
| `result/baseline.txt` | 每个 dataset/split 实时追加的 accuracy 记录 |
| `result/experiment.log` | 每条样本的详细推理日志 |
| `result/preds.jsonl` | 每条样本的预测 + 答案（需加 `--predictions-jsonl`） |

## 完整参数说明

```
python run.py --help
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | 必填 | vllm serve 注册的模型名 |
| `--api-url` | `http://localhost:8000/v1` | vllm serve 地址 |
| `--api-key` | `not-needed` | API Key（vllm 本地部署无需验证） |
| `--model-tag` | 同 `--model` | 结果文件中显示的模型名 |
| `--dataset-root` | `TomDatasets` | 数据集根目录 |
| `--prompt-dir` | `prompt` | prompt 模板目录 |
| `--result-dir` | `result` | 结果输出目录 |
| `--predictions-jsonl` | 不保存 | 每条预测的输出路径 |
| `--max-new-tokens` | `2048` | 最大生成 token 数 |
| `--temperature` | `0.01` | 采样温度 |
| `--top-p` | `0.95` | Nucleus 采样参数 |
| `--dataset-filter` | 全部 | 只评测指定数据集，空格分隔 |
| `--split-filter` | `test` | 只评测指定 split |
| `--include-all-splits` | 关闭 | 评测所有 split（含 train/validation） |
| `--max-samples-per-split` | `0`（全量） | 每个 split 最多取 N 条 |
| `--prompt-names` | 全部 | 只运行指定 prompt 模板，空格分隔 |
| `--prompt-style` | `two_layer` | `two_layer`（MCQ + open）或 `legacy` |
| `--shuffle-repeats` | `5` | MCQ 选项随机排列重复次数 |
| `--summary-columns` | `dataset_split` | 结果表列粒度：`dataset_split` 或 `dataset` |
| `--eval-phase` | `none` | `screen`（自动限 50 条）或 `final`（全量） |

## 项目结构

```
TomTest/
├── run.py          # 主循环：参数解析、批量推理、输出汇总
├── data.py         # 数据加载与 schema 规范化
├── prompt.py       # 模板加载、字段填充、MCQ 选项构造
├── scoring.py      # 答案打分（mcq_letter / open_substring / yes_no）
├── llm/
│   ├── __init__.py
│   └── client.py   # LLMClient，封装 OpenAI 兼容 API + 批量并发 + 自动重试
├── prompt/         # prompt 模板目录（.txt 文件，每个文件一套策略）
│   ├── main_mcq_abcd.txt
│   ├── main_open.txt
│   ├── Standard Zero-shot.txt
│   └── ...
├── TomDatasets/    # 数据集（见上方下载说明）
├── result/         # 评测结果输出（自动创建）
└── requirements.txt
```

## 自定义 Prompt 模板

在 `prompt/` 目录下新建 `.txt` 文件即可自动加入评测，文件名作为模板标识出现在结果表中。

模板内支持以下占位符：

| 占位符 | 内容 |
|---|---|
| `{Story}` | 故事文本 |
| `{Question}` | 问题 |
| `{Action}` | 动作序列（JSON） |
| `{State}` | 状态信息（JSON） |
| `{Meta}` | 元信息（JSON） |
| `{options_block}` | MCQ 选项块（`two_layer` 模式自动填充） |
| `{Story[full_story]}` | Story 的 full_story 子字段 |

## 许可证

MIT License
