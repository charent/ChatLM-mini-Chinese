
import os, PIL, sys 
import numpy as np
from torch.utils.data import DataLoader
import torch 
from torch import nn 
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from transformers import TrainingArguments, Trainer
from tokenizers import Tokenizer

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
from utils.functions import get_bleu4_score


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

        # file_name=None会自动生成log文件名
        self.logger = Logger('chat_trainer', save2file=True, file_name=train_config.trainer_log_file) 
    
    def train(self, ) -> None:
        '''
        '''
        log = self.logger
        train_config = self.train_config
        model_config = self.model_config

        log.info('loading datasets ...')
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

        log.info('train dataset size: {}, validation dataset size {}.'.format(dataset.get_dataset_size('train'), dataset.get_dataset_size('validation')), save_to_file=True)

        accelerator = Accelerator(mixed_precision=train_config.mixed_precision)
        device = accelerator.device
        log.info('device {} is used!'.format(str(device)), save_to_file=True)

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
        
        steps_per_epoch = np.ceil(dataset.get_dataset_size('train') // train_config.batch_size)
        eval_steps = np.ceil(dataset.get_dataset_size('validation') // train_config.batch_size)

        best_bleu4 = 0.0
        best_epoch = 0
        epoch_loss_sum = 0.0
        step_loss_sum = 0.0

        with Progress(TextColumn("[progress.description]{task.description}"),
              BarColumn(),
              TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
              TimeRemainingColumn(),
              TimeElapsedColumn(),
              TextColumn("[bold blue]{task.fields[loss_info]}"),
             ) as progress:
            
            epoch_progress = progress.add_task(description='epoch: ', info='', total=train_config.epochs)
            steps_progress = progress.add_task(description='bacth: ', info='', total=steps_per_epoch)
            eval_progress = progress.add_task(description='evaluate: ', info='', total=eval_steps)

            for epoch in range(train_config.epochs):
                model.train()
                
                epoch_show_txt = 'epoch: {}/{}, avg_loss: {:.6f}, best_epoch: {}, best_bleu: {}'.format(
                    epoch, train_config.epochs, epoch_loss_sum / steps_per_epoch, best_epoch, best_bleu4
                )
                progress.update(epoch_progress, info=epoch_show_txt)
                progress.reset(steps_progress)
                
                for step, batch_data in enumerate(train_dataloader):

                    # 更新进度条
                    step_show_txt = 'step: {}/{}, loss: {:.6f}'.format(step, steps_per_epoch, step_loss_sum // train_config.batch_size)
                    progress.advance(steps_progress, advance=1)
                    progress.update(steps_progress, info=step_show_txt)

                    inputs_ids, inputs_mask = batch_data['inputs_ids'], batch_data['inputs_mask']
                    # target_ids, target_mask = batch_data['target_ids'], batch_data['target_mask']
                    target_ids = batch_data['target_ids']

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

                    break
                
                #  end for
                progress.advance(epoch_progress, advance=1)
                model.eval()
                
                cur_bleu4_score = self.evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    valid_dataloader=valid_dataloader,
                    accelerator=accelerator,
                    eval_progress=eval_progress,
                    progress=progress,
                    )

                accelerator.wait_for_everyone()

                # save model
                if cur_bleu4_score >= best_bleu4:
                    model_dict = accelerator.get_state_dict(model)
                    accelerator.save(model_dict, train_config.model_file.format(epoch))

    def evaluate(self, 
                model: TextToTextModel, 
                tokenizer: Tokenizer,
                valid_dataloader: DataLoader, 
                accelerator: Accelerator,
                eval_progress: Progress,
                progress: Progress,
            ) -> float:
        
        '''
        评估，返回平均的bleu分数
        '''
        progress.reset(eval_progress)
        with torch.no_grad():
            for _, batch_data in enumerate(valid_dataloader):
                progress.advance(eval_progress, advance=1)
               

                inputs_ids, inputs_mask = batch_data['inputs_ids'], batch_data['inputs_mask']
                target_ids = batch_data['target_ids']

                outputs = model.generate(
                    input_ids=inputs_ids,
                    attention_mask=inputs_mask
                )


            # gather data from multi-gpus (used when in ddp mode)
            outputs = accelerator.gather_for_metrics(outputs)
            target_ids = accelerator.gather_for_metrics(target_ids)

            outputs = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
            target = [tokenizer.decode(t, skip_special_tokens=True) for t in target_ids]

    
            progress.update(eval_progress, info='')

        
       

    def test(self, ) -> None:
        '''
        '''
        pass
    
    def print_and_log(self, info: str) -> None:
        print(info)
        self.logger.info(info)

if __name__ == '__main__':
    
    # trainer = ChatTrainer()
    train_config = TrainConfig()
    model_config = T5ModelConfig()

    chat_trainer = ChatTrainer(train_config=train_config, model_config=model_config)

    chat_trainer.train()