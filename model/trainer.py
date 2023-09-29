
import os, PIL, sys 
import numpy as np
from torch.utils.data import DataLoader
import torch 
from torch import nn 

from transformers import TrainingArguments, Trainer

# import accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed

sys.path.append('.')
sys.path.append('..')

# import 自定义类和函数
from model.chat_model import TextToTextModel
from utils.functions import get_bleu4_score
from utils.logger import Logger
from model.chat_dataset import ParquetDataset
from config import PROJECT_ROOT, TrainConfig, T5ModelConfig



def transformers_trainer(config: TrainConfig) -> None:
    ''''
    
    '''
    trainer_args = TrainingArguments()

    model = None 
    train_data = None
    dev_data = None
    tokenizer = None 
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=tokenizer,
        compute_metrics=None,
    )

class ChatTrainer:
    def __init__(self, train_config: TrainConfig, model_config: T5ModelConfig, ) -> None:
        
        self.train_config = train_config
        self.model_config = model_config
        self.logger = Logger('chat_trainer', save2file=True, file_name=train_config.trainer_log_file) # file_name=None会自动生成log文件名
    
    def train(self, ) -> None:
        '''
        '''
        logger = self.logger
        train_config = self.train_config
        model_config = self.model_config

        logger.info('loading datasets ...')
        dataset = ParquetDataset(
            parquet_file={
                'train': train_config.train_file,
                'validation': train_config.validation_file,
            }, 
            tokenizer_file=train_config.tokenizer_file, 
            buffer_size=train_config.dataloader_buffer_size,
            max_len=train_config.max_seq_len,
            seed=train_config.seed,
        )
        
        train_dataloader = DataLoader(dataset['train'], batch_size=train_config.batch_size)
        valid_dataloader = DataLoader(dataset['validation'], batch_size=train_config.batch_size)

        accelerator = Accelerator(mixed_precision=train_config.mixed_precision)
        device = accelerator.device
        logger.info('device {} is used!'.format(str(device)))

        # T5: All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        tokenizer = dataset.tokenizer
        decoder_start_token_id = tokenizer.token_to_id('[PAD]')
        model_config.vocab_size = tokenizer.get_vocab_size()  # 往config添加vocab_size

        model = TextToTextModel(config=model_config, decoder_start_token_id=decoder_start_token_id)

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=train_config.learn_rate)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer, 
                max_lr=25 * train_config.learn_rate, 
                epochs=train_config.epochs, 
                steps_per_epoch=dataset.get_dataset_size('train') # 获取train dataset的长度
                )
        
        model, optimizer, lr_scheduler, train_dataloader, valid_dataloader = accelerator.prepare(
                model, 
                optimizer,
                lr_scheduler, 
                train_dataloader, 
                valid_dataloader
            )
        

        for epoch in range(train_config.epochs):
            model.train()
            for step, batch_data in enumerate(train_dataloader):
                inputs_ids, inputs_mask = batch_data['inputs_ids'], batch_data['inputs_mask']
                target_ids, target_mask = batch_data['target_ids'], batch_data['target_mask']
                
                # for t5 model, all labels set to `-100` are ignored (masked)
                target_ids[target_ids == decoder_start_token_id] = -100

                # print("inputs:{}, mask:{}, target_ids:{}".format(inputs_ids.shape, inputs_mask.shape, target_ids.shape))
                
                outputs = model(
                    input_ids=inputs_ids,
                    input_mask=inputs_mask,
                    labels=target_ids
                )

                loss = outputs.loss

                # attention here! loss.backward()
                accelerator.backward(loss) 
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                
    
    def evaluate(self, ) -> None:
        '''
        '''
        pass

    def test(self, ) -> None:
        '''
        '''
        pass


if __name__ == '__main__':
    
    # trainer = ChatTrainer()
    train_config = TrainConfig()
    model_config = T5ModelConfig()

    chat_trainer = ChatTrainer(train_config=train_config, model_config=model_config)

    chat_trainer.train()