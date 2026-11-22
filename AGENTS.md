# AGENTS.md

供 AI 编码代理（及人类贡献者）使用的项目说明。动手前请先读完本文件。

## 项目概览

ChatLM-mini-Chinese 是一个中文对话 0.2B 小模型（T5 架构），完整覆盖
tokenizer 训练 → Text-to-Text 预训练 → SFT 微调 → DPO 偏好优化 → 推理服务的全链路。

## 环境

- Python `>=3.10,<3.12`（`pyproject.toml` 已限制，`.python-version` 固定 3.10）
- 依赖用 [uv](https://docs.astral.sh/uv/) 管理：`uv sync` 创建 `.venv` 并以 editable 模式安装 `chatlm`
- **`uv.lock` 必须提交**，新增/升级依赖后执行 `uv lock` 一并提交
- `torch` 走 `[tool.uv.index]` 中配置的 CUDA 12.1 官方源，不要手动 `pip install torch`

## 目录结构

```
src/chatlm/           # 全部业务代码（唯一的 Python 包）
  paths.py            # 统一路径（PROJECT_ROOT / DATA_DIR / ARTIFACT_DIR）
  config.py           # 训练与推理配置（dataclass）
  cli.py              # 统一命令行入口 chatlm
  data/               # 数据集、tokenizer 训练、原始语料与 DPO 数据处理
  models/             # TextToTextModel 定义
  training/           # 预训练 / SFT / DPO / accelerate 手写训练循环
  inference/          # ChatBot、交互式 CLI、FastAPI 服务
  utils/              # logging / metrics / text / io / hf / plot
tests/                # pytest 单元测试
configs/              # accelerate 等运行配置
notebook/             # 交互式训练与评测 notebook（train / plot_data / c_eval / cmmlu）
data/                 # 数据集（仅示例 parquet 入库）
artifacts/            # 训练产物：tokenizer、权重、日志（不入库）
docs/img/             # 文档配图
```

## 常用命令

| 场景 | 命令 |
| --- | --- |
| 安装依赖 | `uv sync`（含开发工具用 `uv sync --group dev`） |
| 训练 tokenizer | `uv run chatlm-tokenizer` |
| 预训练（本项目 trainer） | `uv run chatlm train` |
| 预训练（HuggingFace trainer） | `uv run chatlm pretrain` |
| SFT 微调 | `uv run chatlm sft` |
| DPO 偏好优化 / 合并 lora | `uv run chatlm dpo train` / `uv run chatlm dpo merge` |
| 命令行对话 | `uv run chatlm-cli` |
| 启动 API 服务 | `uv run chatlm-api` |
| 多卡训练 | `accelerate launch --multi_gpu --num_processes N -m chatlm.training.sft` |

## 代码约定

- **不要在仓库根目录新增 `.py` 脚本**：新入口放进 `src/chatlm/` 下对应子包，并在
  `pyproject.toml` 的 `[project.scripts]` 与 `src/chatlm/cli.py` 的 `COMMANDS` 中登记。
- **包内一律绝对导入**：`from chatlm.data.dataset import build_seq2seq_dataset`。
  禁止 `from model...`、`from config import ...`；禁止 `sys.path.extend([...])` 之类的 hack。
- **路径统一走 `chatlm.paths`**：禁止 `PROJECT_ROOT + '/xxx'` 字符串拼接，用
  `DATA_DIR / "x.parquet"`、`ARTIFACT_DIR / "sft"` 等 `pathlib` 表达。
- **配置集中在 `chatlm/config.py`**：新增字段用 dataclass + `field(default_factory=...)`，
  保证可被环境变量/测试注入，不在 import 阶段产生副作用。
- **日志用 `chatlm.utils.logging.Logger`**，不要引入 `loguru`，也不要直接 `print` 记录训练信息。
- **入口统一 `main()` + `argparse`**，模块顶层不要有副作用（加载模型、读文件等放 `main()` 内）。
- 训练产物写 `artifacts/`、数据集放 `data/`，二者已被 `.gitignore` 忽略，不要提交。

## 检查与测试

每次改完代码按顺序执行：

```bash
uv run ruff format .
uv run ruff check . --fix
uv run ruff check .
uv run pytest -q
```

- ruff 版本固定在 dev 依赖中（`ruff==0.17.6`），规则与例外理由写在 `pyproject.toml`
  的注释里，**不要为迎合检查而放宽规则**。
- 提交前可选安装钩子：`uv run pre-commit install`（配置见 `.pre-commit-config.yaml`）。
- CI 见 `.github/workflows/ci.yml`：lint（ruff）与 test（pytest）两个 job，不跑训练。

## 注意事项

- `notebook/` 下的 notebook 依赖已安装的 `chatlm` 包，不要再添加 `sys.path` 补丁。
- `artifacts/config.json` 中的 `auto_map` 由 HuggingFace `save_pretrained` 生成，指向同目录的
  `modeling_chat_model.py`，属产物，不要手工编辑或提交。
- 预训练存在两条实现路线（本项目 `accelerate` 手写循环与 HuggingFace `Seq2SeqTrainer`），
  两者行为应保持一致，改动数据集/指标时需同步验证。
- 详细的数据准备、评测与结果说明见 `README.md`。
