"""项目统一路径定义。

所有路径均使用 :class:`pathlib.Path` 表达，避免硬编码 ``/`` 与 ``\\`` 造成的
跨平台问题，也不再依赖「从仓库根目录启动脚本」这一隐含前提。

可通过环境变量覆盖数据与产物根目录：

- ``CHATLM_DATA_DIR``：数据集目录，默认为 ``<repo>/data``
- ``CHATLM_ARTIFACT_DIR``：训练产物目录，默认为 ``<repo>/artifacts``
"""

from __future__ import annotations

import os
from pathlib import Path

# src/chatlm/paths.py -> parents[2] 即仓库根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("CHATLM_DATA_DIR", PROJECT_ROOT / "data")).resolve()
ARTIFACT_DIR = Path(os.getenv("CHATLM_ARTIFACT_DIR", PROJECT_ROOT / "artifacts")).resolve()

# 常用子目录
LOG_DIR = ARTIFACT_DIR / "logs"
# tokenizer 与模型权重放在同一目录，推理时只需一个 model_dir
TOKENIZER_DIR = ARTIFACT_DIR
# 文档配图目录（原根目录 img/）
IMG_DIR = PROJECT_ROOT / "docs" / "img"
PRETRAIN_DIR = ARTIFACT_DIR / "pretrain"
SFT_DIR = ARTIFACT_DIR / "sft"
DPO_DIR = ARTIFACT_DIR / "dpo"
LORA_DIR = ARTIFACT_DIR / "lora"
TRAIN_STATE_DIR = ARTIFACT_DIR / "train_latest_state"


def ensure_artifact_dirs() -> None:
    """创建所有产物子目录（幂等）。"""
    for directory in (
        ARTIFACT_DIR,
        LOG_DIR,
        TOKENIZER_DIR,
        PRETRAIN_DIR,
        SFT_DIR,
        DPO_DIR,
        LORA_DIR,
        TRAIN_STATE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "ARTIFACT_DIR",
    "LOG_DIR",
    "TOKENIZER_DIR",
    "PRETRAIN_DIR",
    "SFT_DIR",
    "DPO_DIR",
    "LORA_DIR",
    "TRAIN_STATE_DIR",
    "ensure_artifact_dirs",
]
