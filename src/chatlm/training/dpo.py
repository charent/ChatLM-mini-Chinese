"""DPO 偏好优化（基于 trl DPOTrainer）。

用法::

    chatlm-dpo train
    chatlm-dpo merge
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType  # pyright: ignore[reportPrivateImportUsage]  # peft 未在 __all__ 中导出
from transformers import PreTrainedTokenizerFast, TrainingArguments
from trl import DPOTrainer

from chatlm.config import DpoConfig, T5ModelConfig
from chatlm.models.chat_model import TextToTextModel
from chatlm.paths import LOG_DIR
from chatlm.utils.hf import get_T5_config

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def build_dpo_dataset(file: str, split: str, cache_dir: str | None = None) -> Dataset:
    """加载 DPO 数据集并转换为所需格式。

    数据集被转换为如下结构::

        {
            'prompt': List[str],
            'chosen': List[str],
            'rejected': List[str],
        }
    """
    dataset = load_dataset("json", data_files=file, split=split, cache_dir=cache_dir)

    def split_prompt_and_responses(sample: dict) -> dict[str, str]:
        return {
            # add an eos token for signal that end of sentence, using in generate.
            "prompt": f"{sample['prompt']}[EOS]",
            "chosen": f"{sample['chosen']}[EOS]",
            "rejected": f"{sample['rejected']}[EOS]",
        }

    return cast(Dataset, dataset.map(split_prompt_and_responses).shuffle(2333))


def _load_sft_model(config: DpoConfig, tokenizer: PreTrainedTokenizerFast) -> TextToTextModel:
    """加载 SFT 权重（目录或单文件）。"""
    if os.path.isdir(config.sft_model_file):
        # 传入文件夹则 from_pretrained
        return cast(TextToTextModel, TextToTextModel.from_pretrained(config.sft_model_file))

    # load_state_dict
    t5_config = get_T5_config(
        T5ModelConfig(),
        vocab_size=len(tokenizer),
        decoder_start_token_id=int(tokenizer.pad_token_id or 0),
        eos_token_id=int(tokenizer.eos_token_id or 0),
    )
    model = TextToTextModel(t5_config)
    model.load_state_dict(torch.load(config.sft_model_file, map_location="cpu"))  # set cpu for no exception
    return model


def train_dpo(config: DpoConfig, peft_config: LoraConfig | None = None) -> None:
    """DPO 训练入口。"""
    # step 1. 加载tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)

    # step 2. 加载预训练模型
    model_train = _load_sft_model(config, tokenizer)
    model_ref = _load_sft_model(config, tokenizer)

    # 4. 加载训练数据集
    train_dataset = build_dpo_dataset(config.dpo_train_file, "train")

    # 5. 加载评估数据集
    # eval_dataset = build_dpo_dataset(config.dpo_eval_file, "train")
    eval_dataset = None

    # 6. 初始化训练参数
    training_args = TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_train_epochs=config.num_train_epochs,
        auto_find_batch_size=True,
        remove_unused_columns=False,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_first_step=True,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        output_dir=config.output_dir,
        optim="adafactor",
        report_to=["tensorboard"],
        log_level="info",
        warmup_steps=config.warmup_steps,
        bf16=False,
        fp16=config.fp16,
        seed=config.seed,
        logging_dir=config.log_dir,
    )

    # 7. 初始化 DPO trainer
    dpo_trainer = DPOTrainer(
        model_train,
        model_ref,
        peft_config=peft_config,  # pyright: ignore[reportArgumentType]  # peft 存根把配置声明为 dict
        args=training_args,
        beta=config.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=config.max_seq_len,
        max_target_length=config.max_seq_len,
        max_prompt_length=config.max_seq_len,
        generate_during_eval=True,
        is_encoder_decoder=True,
    )

    # 8. 训练
    dpo_trainer.train()

    # 9. 保存日志
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    loss_log = pd.DataFrame(dpo_trainer.state.log_history)
    loss_log.to_csv(LOG_DIR / f"dpo_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")

    # 10. 保存模型/lora
    save_subdir = "lora" if peft_config is not None else "dpo"
    model_save_dir = Path(config.sft_model_file).parent / save_subdir

    dpo_trainer.save_model(str(model_save_dir))
    print(f"save model or lora adapter to: {model_save_dir}")


def merge_lora_weight_into_model(config: DpoConfig, peft_config: LoraConfig) -> None:
    """将 lora adapter 合并进 SFT 权重。"""
    # step 1. 加载tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)

    # step 2. 加载预训练模型
    sft_model = _load_sft_model(config, tokenizer)

    # 注意这个路径要和 train_dpo 中保存 lora 的路径一致
    adapter_save_dir = Path(config.sft_model_file).parent / "lora"

    peft_model = PeftModel.from_pretrained(
        model=sft_model,
        model_id=str(adapter_save_dir),
        config=peft_config,
        adapter_name="adapter",
    )

    # 3. load adapter
    print(f"load adapter from dir: {adapter_save_dir}")

    peft_model.load_adapter(str(adapter_save_dir), "adapter")

    # 4. merge
    merged_model = peft_model.merge_and_unload()

    # 5. save
    save_merge_file = config.sft_model_file + ".dpo_lora_merged"
    merged_model.save_pretrained(save_merge_file)
    print(f"save merge model file to: {save_merge_file}")


def build_lora_config() -> LoraConfig:
    """默认 LoRA 配置。"""
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,  # text 2 text lora model
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="all",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatLM DPO 偏好优化")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="训练 DPO")
    subparsers.add_parser("merge", help="合并 lora adapter 到 SFT 权重")
    args = parser.parse_args()

    dpo_config = DpoConfig()

    if args.command == "train":
        train_dpo(dpo_config, peft_config=None)
    else:
        merge_lora_weight_into_model(dpo_config, build_lora_config())


if __name__ == "__main__":
    main()


__all__ = ["build_dpo_dataset", "train_dpo", "merge_lora_weight_into_model", "build_lora_config", "main"]
