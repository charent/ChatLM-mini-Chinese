"""SFT 微调（基于 HuggingFace Seq2SeqTrainer）。

用法::

    chatlm-sft
    accelerate launch --multi_gpu --num_processes 2 -m chatlm.training.sft
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    GenerationConfig,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from chatlm.config import SFTConfig, T5ModelConfig
from chatlm.data.dataset import build_seq2seq_dataset
from chatlm.models.chat_model import TextToTextModel
from chatlm.paths import LOG_DIR
from chatlm.utils.hf import MyTrainerCallback, get_T5_config

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def sft_train(config: SFTConfig, resume_from_checkpoint: bool = False) -> None:
    """SFT 训练入口。"""
    # step 1. 加载tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)

    # step 2. 加载预训练模型
    if os.path.isdir(config.finetune_from_ckp_file):
        # 传入文件夹则 from_pretrained
        model = TextToTextModel.from_pretrained(config.finetune_from_ckp_file)
    else:
        # load_state_dict
        t5_config = get_T5_config(
            T5ModelConfig(),
            vocab_size=len(tokenizer),
            decoder_start_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = TextToTextModel(t5_config)
        model.load_state_dict(torch.load(config.finetune_from_ckp_file, map_location="cpu"))  # set cpu for no exception

    # Step 4: 加载数据集
    dataset = build_seq2seq_dataset(file=config.sft_train_file, split="train", tokenizer=tokenizer, file_format="json")

    # Step 5: 定义训练参数
    # T5属于sequence to sequence模型，故要使用Seq2SeqTrainingArguments、DataCollatorForSeq2Seq、Seq2SeqTrainer
    generation_config = GenerationConfig()
    generation_config.remove_invalid_values = True
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.decoder_start_token_id = tokenizer.pad_token_id
    generation_config.max_new_tokens = 320
    generation_config.repetition_penalty = 1.5
    generation_config.num_beams = 1  # greedy search
    generation_config.do_sample = False  # greedy search

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        auto_find_batch_size=True,  # 防止OOM
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        num_train_epochs=config.num_train_epochs,
        optim="adafactor",
        report_to="tensorboard",
        log_level="info",
        save_steps=config.save_steps,
        save_total_limit=3,
        fp16=config.fp16,
        logging_first_step=config.logging_first_step,
        warmup_steps=config.warmup_steps,
        seed=config.seed,
        generation_config=generation_config,
    )

    # step 6: 初始化 collator
    collator = DataCollatorForSeq2Seq(tokenizer, max_length=config.max_seq_len)
    empty_cuda_cache = MyTrainerCallback()

    # Step 7: 定义 Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[empty_cuda_cache],
    )

    # step 8: 训练
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # step 9: 保存日志
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    loss_log = pd.DataFrame(trainer.state.log_history)
    loss_log.to_csv(LOG_DIR / f"sft_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")

    # Step 10: 保存模型
    trainer.save_model(config.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatLM SFT 微调")
    parser.add_argument("--resume-from-checkpoint", action="store_true", help="从断点处继续训练")
    args = parser.parse_args()

    sft_train(SFTConfig(), resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()


__all__ = ["sft_train", "main"]
