# coding=utf-8
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerFast, TrainingArguments
from trl import DPOTrainer
from tokenizers import Tokenizer
from peft import LoraConfig, TaskType, PeftModel

from config import DpoConfig
from model.chat_model import TextToTextModel
from utils.functions import json_to_dataclass
from utils.logger import Logger

log = Logger('train_dpo', save2file=True)

def get_dataset(split: str, file: str, cache_dir: str = None) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_dataset('json', data_files=file,  split=split, cache_dir=cache_dir)

    def split_prompt_and_responses(sample: dict) -> Dict[str, str]:
        return {
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": sample["reject"],
        }

    return dataset.map(split_prompt_and_responses)


def train_dpo(config: DpoConfig, peft_config: LoraConfig=None) -> None:

    # 0. 加载模型配置文件
    model_config_class = json_to_dataclass(config.model_config_file, 'ModelConfig')
    model_config = model_config_class()

    # 1. 加载tokenizer
    tokenizer_obj = Tokenizer.from_file(config.tokenizer_file)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
    tokenizer.pad_token = '[PAD]'
    tokenizer.pad_token_id = tokenizer_obj.token_to_id('[PAD]')
    tokenizer.unk_token = '[UNK]'
    tokenizer.unk_token_id = tokenizer_obj.token_to_id('[UNK]')
    tokenizer.eos_token = '[EOS]'
    tokenizer.eos_token_id = tokenizer_obj.token_to_id('[EOS]')
    
    # 2. 加载预训练模型
    model = TextToTextModel(config=model_config, decoder_start_token_id=tokenizer.pad_token_id)
    model.load_state_dict(torch.load(config.sft_model_file))
    model_train = model.model # un_pkg model

    model_ref = TextToTextModel(config=model_config, decoder_start_token_id=tokenizer.pad_token_id)
    model_ref.load_state_dict(torch.load(config.sft_model_file))
    model_ref = model_ref.model
    
    # 3. 加载训练数据集
    train_dataset = get_dataset("train", file=config.dpo_train_file)

    # 4. 加载评估数据集
    eval_dataset = get_dataset("train", file=config.dpo_eval_file)

    # 5. 初始化训练参数
    training_args = TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        max_steps=config.max_steps,
        remove_unused_columns=False,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=config.logging_steps,  # match results in blog post
        eval_steps=config.eval_steps,
        output_dir=config.output_dir,
        optim="rmsprop",
        warmup_steps=config.warmup_steps,
        bf16=False,
        fp16=config.fp16,
        seed=config.seed
    )

    # 6. 初始化 DPO trainer
    dpo_trainer = DPOTrainer(
        model_train,
        model_ref,
        peft_config=peft_config,
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

    log.info('dpo train agrs: \n{}\n'.format(training_args), save_to_file=True)
    log.info('dpo train peft config: \n{}\n'.format(peft_config), save_to_file=True)

    # 7. 训练
    dpo_trainer.train()

    # 8. 保存模型/lora
    suffixe = '/lora/' if peft_config is not None else ''
    model_save_dir = '/'.join(config.sft_model_file.split('/')[0: -1]) + suffixe

    dpo_trainer.save_model(model_save_dir)
    print('save model or lora adapter to: {}'.format(model_save_dir))

    dpo_trainer.accelerator.wait_for_everyone()
    model.model =  dpo_trainer.model
    
    if peft_config is None:
        sate_dict_dir = config.sft_model_file + '.dpo.lora.bin'
        torch.save(model.state_dict(), sate_dict_dir)
        print('save sate dict to: {}'.format(sate_dict_dir))

def merge_lora_weight_into_model(config: DpoConfig, peft_config: LoraConfig) -> None:

     # 0. 加载模型配置文件
    model_config_class = json_to_dataclass(config.model_config_file, 'ModelConfig')
    model_config = model_config_class()

    # 1. 加载tokenizer
    tokenizer_obj = Tokenizer.from_file(config.tokenizer_file)
    
    # 2. 加载预训练模型
    model = TextToTextModel(config=model_config, decoder_start_token_id=tokenizer_obj.token_to_id('[PAD]'))
    model.load_state_dict(torch.load(config.sft_model_file))
    model_sft = model.model # un package model

    adapter_save_dir = '/'.join(config.sft_model_file.split('/')[0: -1])
    # peft_model = PeftModel.from_pretrained(
    #     model=model_sft,
    #     model_id=model_save_dir,
    #     config=peft_config,
    #     adapter_name='adapter',
    # )
    
    peft_model = PeftModel(
        model=model_sft,
        peft_config=peft_config,
        adapter_name='adapter',
    )

    # 3. load adapter
    
    print('load adapter from dir: {}'.format(adapter_save_dir))

    peft_model.load_adapter(model_id=adapter_save_dir, adapter_name='adapter',)

    # 4. merge
    peft_model = peft_model.merge_and_unload()

    model.model = peft_model # package model
    
    # 5. save
    save_merge_file = config.sft_model_file + '.dpo.lora_merged.bin'
    torch.save(model.state_dict(), save_merge_file)
    print('save merge model file to: {}'.format(save_merge_file))

   
if __name__ == "__main__":

    peft_config = LoraConfig(
         task_type=TaskType.SEQ_2_SEQ_LM,  # text 2 text lora model 
         inference_mode=False, 
         r=16, 
         lora_alpha=16, 
         lora_dropout=0.1, 
         bias="all",
    )

    dpo_config = DpoConfig()

    # 1. train
    train_dpo(dpo_config, None)

    # 2. merge lora adapter into model
    # merge_lora_weight_into_model(dpo_config, peft_config)




    