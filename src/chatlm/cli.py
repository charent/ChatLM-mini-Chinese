"""统一命令行入口。

用法::

    chatlm train                 # accelerate 手写训练循环（预训练）
    chatlm pretrain              # HuggingFace Seq2SeqTrainer 预训练
    chatlm sft                   # SFT 微调
    chatlm dpo train             # DPO 偏好优化
    chatlm dpo merge             # 合并 lora adapter
    chatlm tokenizer             # 训练 tokenizer
    chatlm cli                   # 命令行对话
    chatlm api                   # 启动 HTTP 服务

子命令后的参数会原样透传给对应模块。
"""

from __future__ import annotations

import argparse
import importlib
import sys

# 子命令 -> (模块, 入口函数)
COMMANDS: dict[str, tuple[str, str]] = {
    "train": ("chatlm.training.accelerate_trainer", "main"),
    "pretrain": ("chatlm.training.pretrain", "main"),
    "sft": ("chatlm.training.sft", "main"),
    "dpo": ("chatlm.training.dpo", "main"),
    "tokenizer": ("chatlm.data.tokenizer", "main"),
    "cli": ("chatlm.inference.cli", "main"),
    "api": ("chatlm.inference.api", "main"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="chatlm",
        description="ChatLM-mini-Chinese 统一命令行入口",
    )
    parser.add_argument("command", choices=sorted(COMMANDS), help="要执行的子命令")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="透传给子命令的参数")
    args = parser.parse_args()

    module_name, func_name = COMMANDS[args.command]
    module = importlib.import_module(module_name)

    # 让子命令自己的 argparse 拿到正确参数
    sys.argv = [f"chatlm {args.command}", *args.args]
    getattr(module, func_name)()


if __name__ == "__main__":
    main()


__all__ = ["COMMANDS", "main"]
