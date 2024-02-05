# coding=utf-8
import time
import os 
import pandas as pd 
from dataclasses import dataclass
import torch
from typing import Dict

from tqdm import tqdm
import numpy as np
from transformers import PreTrainedTokenizerFast, Seq2SeqTrainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

from transformers.generation.configuration_utils import GenerationConfig
from datasets import Dataset, load_dataset

from model.chat_model import TextToTextModel
from model.dataset import MyDataset
from config import TrainConfig, T5ModelConfig

from utils.functions import json_to_dataclass, get_T5_config, MyTrainerCallback

tqdm.pandas()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def get_dataset(file: str, split: str, tokenizer: PreTrainedTokenizerFast,  cache_dir: str='.cache') -> Dataset:
    """
    加载数据集
    """
    dataset = load_dataset('parquet', data_files=file,  split=split, cache_dir=cache_dir)

    def tokens_to_ids(samples: dict) -> Dict[str, str]:

        eos_token_id = tokenizer.eos_token_id

        batch_prompt = samples['prompt']
        batch_response = samples['response']

        encoded_prompt = tokenizer(batch_prompt, truncation=False, padding=False, return_attention_mask=False,)
        encoded_response = tokenizer(batch_response, truncation=False, padding=False, return_attention_mask=False,)

        # vocab size 小于65535 可以用 uint16, 每个样本都要添加eos_token_id
        input_ids = [np.array(item + [eos_token_id], dtype=np.uint16) for item in encoded_prompt["input_ids"]]
        labels = [np.array(item + [eos_token_id], dtype=np.uint16) for item in encoded_response["input_ids"]]

        return {
            'input_ids': input_ids,
            'labels': labels,
        }

    dataset = dataset.map(tokens_to_ids, batched=True, batch_size=8192, remove_columns=dataset.column_names)

    return dataset

def pre_train(config: TrainConfig) -> None:

    # step 1. 加载tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)
    
    # step 2. 加载模型配置文件
    t5_config = get_T5_config(T5ModelConfig(), vocab_size=len(tokenizer), decoder_start_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    
    # step 3. 初始化模型
    model = TextToTextModel(t5_config)

    # Step 4: Load my dataset
    dataset = get_dataset(file=config.train_file, split='train', tokenizer=tokenizer)

    # Step 5: Define the training arguments

    # T5属于sequence to sequence模型，故要使用Seq2SeqTrainingArguments、DataCollatorForSeq2Seq、Seq2SeqTrainer
    # huggingface官网的sft工具适用于language model/LM模型

    generation_config = GenerationConfig()
    generation_config.remove_invalid_values = True
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.decoder_start_token_id = tokenizer.pad_token_id
    generation_config.max_new_tokens = 320
    generation_config.num_beams = 1         # greedy search
    generation_config.do_sample = False     # greedy search

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size_per_gpu,
        auto_find_batch_size=True,  # 防止OOM
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learn_rate,
        logging_steps=config.logging_steps,
        num_train_epochs=config.epochs,
        optim="adafactor",
        report_to='tensorboard',
        log_level='info',
        save_steps=config.save_steps,
        save_total_limit=3,
        fp16=True if config.mixed_precision == 'fp16' else False,
        bf16=True if config.mixed_precision == 'bf16' else False,
        logging_first_step=True,
        warmup_steps=config.warmup_steps,
        seed=config.seed,
        generation_config=generation_config,
    )

    # step 6: init my collator,
    collator = DataCollatorForSeq2Seq(tokenizer, max_length=config.max_seq_len)
    empty_cuda_cahce = MyTrainerCallback()

    # Step 7: Define the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[empty_cuda_cahce],
    )

    # step 8: train
    trainer.train(
        # resume_from_checkpoint=True
    )

    #step 9: save log
    loss_log = pd.DataFrame(trainer.state.log_history)
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    loss_log.to_csv(f"{log_dir}/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")

    # Step 10: Save the model
    trainer.save_model(config.output_dir)


if __name__ == '__main__':
    config = TrainConfig()
    pre_train(config)