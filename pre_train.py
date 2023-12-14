# coding=utf-8
import time 
import pandas as pd 
from dataclasses import dataclass

from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, Seq2SeqTrainer, DataCollatorForSeq2Seq,Seq2SeqTrainingArguments
from transformers.generation.configuration_utils import GenerationConfig

from model.chat_model import TextToTextModel
from model.dataset import MyDataset
from config import TrainConfig
from utils.functions import json_to_dataclass

tqdm.pandas()

@dataclass
class My_DataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        '''
        将文本编码为id，MyDataset的`__getitem__`方法返回的是: (str, str)
        features:list[tuple[str, str]]
        '''    
        prompt = [item[0] for item in features]
        resopnse = [item[1] for item in features]

        tokenizer =self.tokenizer
        prompt_encoded = tokenizer(prompt, padding=False, return_token_type_ids=False, return_attention_mask=False)['input_ids']
        resopnse_encoded = tokenizer(resopnse, padding=False, return_token_type_ids=False, return_attention_mask=False)['input_ids']

        batch_size = len(features)
        data = []
        for i in range(batch_size):
            data.append(
                {
                    'input_ids': prompt_encoded[i],
                    'labels': resopnse_encoded[i]
                }
            )

        return super().__call__(data, return_tensors)
    

def pre_train(config: TrainConfig) -> None:

    # step 1. 加载模型配置文件
    model_config_class = json_to_dataclass(config.model_config_file, 'ModelConfig')
    model_config = model_config_class()

    # step 2. 加载tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)
    
    # step 3. 初始化模型
    model = TextToTextModel(config=model_config, decoder_start_token_id=tokenizer.pad_token_id)
    model = model.model

    # Step 4: Load my dataset
    dataset = MyDataset(
        parquet_file=config.train_file,
        tokenizer_dir=config.tokenizer_dir,
        buffer_size=40960,
    )

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
    collator = My_DataCollatorForSeq2Seq(tokenizer, max_length=config.max_seq_len)
    
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

    #step 9: save log
    loss_log = pd.DataFrame(trainer.state.log_history)
    loss_log.to_csv(f"./logs/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")

    # Step 10: Save the model
    trainer.save_model(config.output_dir)


if __name__ == '__main__':
    config = TrainConfig()
    pre_train(config)