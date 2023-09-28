
import os, PIL, sys 
import numpy as np
from torch.utils.data import DataLoader, Dataset
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

from config import TrainConfig, ModelConfig



def transformers_trainer(config: dict) -> None:
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
    def __init__(self, train_config: TrainConfig, model_config: ModelConfig) -> None:
        
        self.config = train_config

        self.logger = Logger('chat_trainer', save2file=True, file_name=train_config['log_file'])

        self.model = TextToTextModel(config=model_config)
    
    def train(self, ) -> None:

        pass
    
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
    model_config = ModelConfig()
    print(model_config)