# coding=utf-8
from typing import Dict
import time 
import os 
import pandas as pd 

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, Seq2SeqTrainer, DataCollatorForSeq2Seq,Seq2SeqTrainingArguments
from transformers.generation.configuration_utils import GenerationConfig

from model.chat_model import TextToTextModel
from config import SFTconfig, T5ModelConfig
from utils.functions import get_T5_config

tqdm.pandas()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def get_dataset(file: str, split: str, encode_fn: callable, encode_args: dict,  cache_dir: str='.cache') -> Dataset:
    """
    Load a dataset
    """
    dataset = load_dataset('json', data_files=file,  split=split, cache_dir=cache_dir)

    def merge_prompt_and_responses(sample: dict) -> Dict[str, str]:
        # add an eos token note that end of sentence, using in generate.
        prompt = encode_fn(sample['prompt'] + '[EOS]', **encode_args)
        response = encode_fn(sample['response'] + '[EOS]', **encode_args)
        return {
            'input_ids': prompt.input_ids,
            'input_mask': prompt.attention_mask,
            'labels': response.input_ids,
        }

    dataset = dataset.map(merge_prompt_and_responses)
    return dataset


def sft_train(config: SFTconfig) -> None:

    # step 1. 加载tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)
    
    # step 2. 加载预训练模型
    model = None
    if os.path.isdir(config.finetune_from_ckp_file):
        # 传入文件夹则 from_pretrained
        model = TextToTextModel.from_pretrained(config.finetune_from_ckp_file)
    else:
        # load_state_dict
        t5_config = get_T5_config(T5ModelConfig(), vocab_size=len(tokenizer), decoder_start_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        model = TextToTextModel(t5_config)
        model.load_state_dict(torch.load(config.finetune_from_ckp_file, map_location='cpu')) # set cpu for no exception

    # Step 4: Load the dataset
    encode_args = {
        'truncation': False,
        'padding': 'max_length',
    }

    dataset = get_dataset(file=config.sft_train_file, encode_fn=tokenizer.encode_plus, encode_args=encode_args, split="train")

    # Step 5: Define the training arguments
    # T5属于sequence to sequence模型，故要使用Seq2SeqTrainingArguments、DataCollatorForSeq2Seq、Seq2SeqTrainer
    # huggingface官网的sft工具适用于language model/LM模型
    generation_config = GenerationConfig()
    generation_config.remove_invalid_values = True
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.decoder_start_token_id = tokenizer.pad_token_id
    generation_config.max_new_tokens = 320
    generation_config.repetition_penalty = 1.5
    generation_config.num_beams = 1         # greedy search
    generation_config.do_sample = False     # greedy search

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        auto_find_batch_size=True,  # 防止OOM
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        num_train_epochs=config.num_train_epochs,
        optim="adafactor",
        report_to='tensorboard',
        log_level='info',
        save_steps=config.save_steps,
        save_total_limit=3,
        fp16=config.fp16,
        logging_first_step=config.logging_first_step,
        warmup_steps=config.warmup_steps,
        seed=config.seed,
        generation_config=generation_config,
    )

    # step 6: init a collator
    collator = DataCollatorForSeq2Seq(tokenizer, max_length=config.max_seq_len)
    
    # Step 7: Define the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # step 8: train
    trainer.train(
        # resume_from_checkpoint=True
    )

    loss_log = pd.DataFrame(trainer.state.log_history)
    loss_log.to_csv(f"./logs/sft_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")

    # Step 9: Save the model
    trainer.save_model(config.output_dir)

if __name__ == '__main__':
    config = SFTconfig()
    sft_train(config)