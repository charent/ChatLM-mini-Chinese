import signal
import sys
import time

from psutil import virtual_memory
import numpy as np
from torch.utils.data import DataLoader
import torch 
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from transformers import TrainingArguments, Trainer
from tokenizers import Tokenizer

# import accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed

# import 自定义类和函数
from model.chat_model import TextToTextModel
from utils.logger import Logger
from model.dataset import MyDataset
from config import PROJECT_ROOT, TrainConfig, T5ModelConfig
from utils.functions import get_bleu4_score, save_model_config, get_free_space_of_disk, my_average


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

        # file_name=None会自动生成以当前日期命名的log文件名
        self.logger = Logger('chat_trainer', std_out=True, save2file=True, file_name=None)

        self.model = None
        self.accelerator = None

        signal.signal(signal.SIGINT, self.process_exit_handler)
    
    def process_exit_handler(self, signal_received, frame) -> None:
        '''
        进程退出时的操作，保存模型
        '''
        ask = "are you sure to exit this process?  Yes (y) or No (n)"
        if self.accelerator:
            self.accelerator.print(ask)
        else:
            print(ask)

        ins = input()
        
        if ins.lower() in ('yes', 'y'):
            if self.accelerator:
                suffix =  'exit_save_{}'.format(str(time.strftime('%Y%m%d%H%M%S', time.localtime())))
                self.accelerator.wait_for_everyone()
                self.save_model(suffix)

                self.accelerator.print('model ckeck point has been saved!')
            else:
                print('process not in trainingg, exit.')
            sys.exit(0)
        else:
            print('do nothing!')

    def save_model(self, suffix: str) -> None:
        if self.model and self.accelerator:
            unwrap_model = self.accelerator.unwrap_model(self.model)
            model_dict =  self.accelerator.get_state_dict(unwrap_model)
            torch.save(model_dict, self.train_config.model_file.format(suffix))
    
    def train(self, ) -> None:
        '''
        '''
        log = self.logger
        train_config = self.train_config
        model_config = self.model_config

        # 根据剩余内存大小决定是否完全加载数据集到内存中
        unuse_mem = virtual_memory().available / (1024 ** 3)  # 单位：GB
        unuse_disk = get_free_space_of_disk('./')

        # 剩余内存≥24GB将把数据集留在内存中
        keep_in_memory = True if unuse_mem >= 24.0 else False

        log.info('cpu memory available: {:.2f} GB, disk space available: {:.2f} GB, keep dataset in memory: {}.'\
                 .format(unuse_mem, unuse_disk, keep_in_memory), save_to_file=True)
        log.info('loading datasets ...')

        train_dataset = MyDataset(
            parquet_file=train_config.train_file,
            tokenizer_file=train_config.tokenizer_file,
            keep_in_memory=keep_in_memory,
            max_seq_len=train_config.max_seq_len,
        )
        valid_dataset = MyDataset(
            parquet_file=train_config.validation_file,
            tokenizer_file=train_config.tokenizer_file,
            keep_in_memory=keep_in_memory,
            max_seq_len=train_config.max_seq_len,
        )

        batch_size = train_config.batch_size_per_gpu
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size,  
            shuffle=False,
            collate_fn=train_dataset.collate_fn,
        )
        valid_dataloader = DataLoader(
            valid_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=valid_dataset.collate_fn,
        )

        log.info('train dataset size: {}, validation dataset size: {}.'\
                .format(len(train_dataset), len(valid_dataset)), save_to_file=True)

        set_seed(train_config.seed)
        accelerator = Accelerator(mixed_precision=train_config.mixed_precision)
        device = accelerator.device
        log.info('using device: {} '.format(str(device)), save_to_file=True)

        # T5: All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        tokenizer = train_dataset.tokenizer
        decoder_start_token_id = tokenizer.token_to_id('[PAD]')
        model_config.vocab_size = tokenizer.get_vocab_size()  # 往config添加vocab_size

        model = TextToTextModel(config=model_config, decoder_start_token_id=decoder_start_token_id)

        # 保存模型配置，方便修改配置后恢复
        save_model_config(model.t5_config.to_diff_dict(), train_config.model_config_file)

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=train_config.learn_rate)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer, 
                max_lr=25 * train_config.learn_rate, 
                epochs=train_config.epochs, 
                steps_per_epoch=len(train_dataset),  # 获取train dataset的长度
                div_factor=train_config.div_factor,
            )
        
        model, optimizer, lr_scheduler, train_dataloader, valid_dataloader = accelerator.prepare(
                model, 
                optimizer,
                lr_scheduler, 
                train_dataloader, 
                valid_dataloader,
            )
        
        self.model = model
        self.accelerator = accelerator

        # 获取当前运行使用了多少个GPU
        num_gpus_used = accelerator.state.num_processes

        # 单机多卡，每个step总共的batch_size = batch_size_per_gpu * num_gpus_used
        # total_batch_size 初始化为batch_size_per_gpu真的只有CPU的情况
        total_batch_size = train_config.batch_size_per_gpu
        if num_gpus_used >= 1:
            total_batch_size = num_gpus_used * train_config.batch_size_per_gpu

        steps_per_epoch = int(np.ceil(len(train_dataset) // total_batch_size))
        eval_steps = int(np.ceil(len(valid_dataset) // total_batch_size))

        best_bleu4 = 0.0
        best_epoch = 0
        epoch_loss_list = []
        step_loss_list = []

        log_loss_interval_n = 50 # 每间隔 n 步保存一次loss到文件

        with Progress(TextColumn("[progress.description]{task.description}"),
              BarColumn(),
              TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
              TimeRemainingColumn(),
              TimeElapsedColumn(),
              TextColumn("[bold blue]{task.fields[show_info]}"),
             ) as progress:
            
            if accelerator.is_main_process:
                epoch_progress = progress.add_task(description='epoch: ', show_info='', total=train_config.epochs)
                steps_progress = progress.add_task(description='steps: ', show_info='', total=steps_per_epoch)
                eval_progress = progress.add_task(description='evaluate: ', show_info='', total=eval_steps, visible=False)

            for epoch in range(train_config.epochs):
                
                if accelerator.is_main_process:
                    epoch_show_txt = 'epoch: {}/{}, avg_loss: {:.6f}, best_epoch: {}, best_bleu: {}'.format(
                        epoch, train_config.epochs, my_average(epoch_loss_list), best_epoch, best_bleu4
                    )
                    progress.update(epoch_progress, show_info=epoch_show_txt)
                    progress.reset(steps_progress)

                epoch_loss_list = []
                model.train()
                
                for step, batch_data in enumerate(train_dataloader):
                    if batch_data is None:
                        print('epoch:{}, step:{}'.format(epoch, step))
                        continue

                    input_ids, input_mask = batch_data['input_ids'], batch_data['input_mask']
                    # target_ids, target_mask = batch_data['target_ids'], batch_data['target_mask']
                    target_ids = batch_data['target_ids']

                    # for t5 model, all labels set to `-100` are ignored (masked)
                    target_ids[target_ids == decoder_start_token_id] = -100

                    # print("inputs:{}, mask:{}, target_ids:{}".format(input_ids.shape, input_mask.shape, target_ids.shape))
                    
                    outputs = model(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        labels=target_ids
                    )

                    loss = outputs.loss.mean()
                    loss_cpu = loss.detach().cpu().numpy()
                    
                    step_loss_list.append(loss_cpu)
                    epoch_loss_list.append(loss_cpu)

                    # attention here! loss.backward()
                    accelerator.backward(loss) 
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                   
                    # 更新进度条
                    if accelerator.is_main_process:
                        step_show_txt = 'step: {}/{}, loss: {:.6f}'.format(step, steps_per_epoch, loss_cpu)
                        progress.advance(steps_progress, advance=1)
                        progress.update(steps_progress, show_info=step_show_txt)

                    # 保存 loss 到文件
                    if step % log_loss_interval_n == 0 or step == eval_steps - 1:
                        
                        info_txt = 'training loss: epoch:{}, step:{}, loss:{}, device:{}'.\
                            format(epoch, step, my_average(step_loss_list), str(accelerator.device))
                        
                        log.info(info_txt, std_out=False, save_to_file=True)
                        step_loss_list = []
                    
                    # if step >= 20:break
                
                #  end for

                #  end for batch
                progress.advance(epoch_progress, advance=1)
                model.eval()
                
                cur_bleu4_score = self.evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    valid_dataloader=valid_dataloader,
                    accelerator=accelerator,
                    eval_progress=eval_progress,
                    eval_steps=eval_steps,
                    progress=progress,
                    )

                # save model
                if cur_bleu4_score >= best_bleu4:
                    best_bleu4 = cur_bleu4_score
                    best_epoch = epoch

                    accelerator.wait_for_everyone()
                    
                    if accelerator.is_main_process: 
                        unwrap_model = accelerator.unwrap_model(model)
                        model_dict = accelerator.get_state_dict(unwrap_model)
                        torch.save(model_dict, train_config.model_file.format(epoch))

                # 每个epoch打印一下日志
                if accelerator.is_main_process:
                    info_txt = 'epoch log: epoch:{}, avg_loss:{}, cur_bleu4:{}, best_bleu4:{}, best_epoch:{}'.\
                                format(epoch, my_average(epoch_loss_list), cur_bleu4_score, best_bleu4, best_epoch)
                    # log.info(info_txt, std_out=True, save_to_file=True)
                    self.print_and_log(info_txt, accelerator)
                    epoch_loss_list = []


    def evaluate(self, 
                model: TextToTextModel, 
                tokenizer: Tokenizer,
                valid_dataloader: DataLoader, 
                accelerator: Accelerator,
                eval_progress: Progress,
                progress: Progress,
                eval_steps: int,
            ) -> float:
        
        '''
        评估，返回平均的bleu分数
        '''
        max_seq_len = self.train_config.max_seq_len
        decode_batch = tokenizer.decode_batch
        bleu4_scores = []

        if accelerator.is_main_process:
            progress.reset(eval_progress)
            progress.update(eval_progress, visible=True)

        with torch.no_grad():
            for step, batch_data in enumerate(valid_dataloader):
                
                if accelerator.is_main_process:
                    progress.advance(eval_progress, advance=1)
                    progress.update(eval_progress, show_info='step: {}/{}'.format(step, eval_steps))

                input_ids, input_mask = batch_data['input_ids'], batch_data['input_mask']
                target_ids = batch_data['target_ids']

                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    max_seq_len=max_seq_len,
                )

      
                # gather data from multi-gpus (used when in ddp mode)
                outputs = accelerator.gather_for_metrics(outputs).cpu().numpy()
                target_ids = accelerator.gather_for_metrics(target_ids).cpu().numpy()
        
                outputs = decode_batch(outputs,  skip_special_tokens=True)
                target_ids = decode_batch(target_ids, skip_special_tokens=True )


                # 删除decode出来字符间的空格
                outputs = [sentance.replace(' ', '') for sentance in outputs]
                target_ids = [sentance.replace(' ', '') for sentance in target_ids]

                # print(outputs, target_ids)

                bleu4_scores = [get_bleu4_score(reference=target_ids[i], outputs=outputs[i]) for i in range(len(target_ids))]
                bleu4_scores.extend(bleu4_scores)

                # if step >= 10: break
        
        avg_bleu4_score = my_average(bleu4_scores)
        if accelerator.is_main_process:
            progress.update(eval_progress, show_info='bleu4 score: {}'.format(avg_bleu4_score))
            progress.update(eval_progress, visible=False)

        return avg_bleu4_score

    def test(self, best_epoch: int=0) -> None:
        '''
        '''
        train_config = self.train_config
        model_config = self.model_config
        log = self.logger

        test_dataset = MyDataset(
            parquet_file=train_config.train_file,
            tokenizer_file=train_config.tokenizer_file,
            keep_in_memory=True,
            max_seq_len=train_config.max_seq_len,
        )
        
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=train_config.batch_size_per_gpu,
            shuffle=False,
            collate_fn=test_dataset.collate_fn
        )

        log.info('test dataset size: {}.'.format(len(test_dataset)), save_to_file=True)

        set_seed(train_config.seed)
        accelerator = Accelerator(mixed_precision=train_config.mixed_precision)
        device = accelerator.device
        log.info('using device: {} '.format(str(device)), save_to_file=True)

         # 获取当前运行使用了多少个GPU
        num_gpus_used = accelerator.state.num_processes

        # 单机多卡，每个step总共的batch_size = batch_size_per_gpu * num_gpus_used
        # total_batch_size 初始化为batch_size_per_gpu真的只有CPU的情况
        total_batch_size = train_config.batch_size_per_gpu
        if num_gpus_used >= 1:
            total_batch_size = num_gpus_used * train_config.batch_size_per_gpu

        # T5: All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        tokenizer = test_dataset.tokenizer
        decoder_start_token_id = tokenizer.token_to_id('[PAD]')
        model_config.vocab_size = tokenizer.get_vocab_size()  # 往config添加vocab_size

        model = TextToTextModel(config=model_config, decoder_start_token_id=decoder_start_token_id)
        model.load_state_dict(torch.load(train_config.model_file.format(best_epoch)))
       
        model, test_dataloader = accelerator.prepare(
                model, 
                test_dataloader,
            )
        
        steps = int(np.ceil(len(test_dataset) // total_batch_size))

        bleu4 = 0.0
        bleu4_scores = []
        decode_batch = tokenizer.decode_batch
        max_seq_len = self.train_config.max_seq_len
        model.eval()

        with Progress(TextColumn("[progress.description]{task.description}"),
              BarColumn(),
              TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
              TimeRemainingColumn(),
              TimeElapsedColumn(),
              TextColumn("[bold blue]{task.fields[show_info]}"),
             ) as progress:
            
            if accelerator.is_main_process:
                steps_progress = progress.add_task(description='steps: ', show_info='', total=steps)
            
            with torch.no_grad():
                for step, batch_data in enumerate(test_dataloader):

                    if accelerator.is_main_process:
                        progress.advance(steps_progress, advance=1)
                        progress.update(steps_progress, show_info='step: {}/{}'.format(step, steps))

                    input_ids, input_mask = batch_data['input_ids'], batch_data['input_mask']
                    target_ids = batch_data['target_ids']

                    # s = time.time()
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=input_mask,
                        max_seq_len=max_seq_len,
                    )
                    # accelerator.print('generate used: {}'.format(time.time() - s))

                    # gather data from multi-gpus (used when in ddp mode)
                    outputs = accelerator.gather_for_metrics(outputs).cpu().numpy()
                    target_ids = accelerator.gather_for_metrics(target_ids).cpu().numpy()
                    
                    outputs = decode_batch(outputs,  skip_special_tokens=True)
                    target_ids = decode_batch(target_ids, skip_special_tokens=True )

                    # 删除decode出来字符间的空格
                    outputs = [sentance.replace(' ', '') for sentance in outputs]
                    target_ids = [sentance.replace(' ', '') for sentance in target_ids]

                    # print('outputs: {}'.format(outputs[0:5]))
                    # print('target_ids: {}'.format(target_ids[0:5]))
                    # print()


                    bleu4_scores = [get_bleu4_score(reference=target_ids[i], outputs=outputs[i]) for i in range(len(target_ids))]
                    bleu4_scores.extend(bleu4_scores)

                    # if step >= 10: break
        
        avg_bleu4_score = my_average(bleu4_scores)
        if accelerator.is_main_process:
            progress.update(steps_progress, show_info='bleu4 score: {}'.format(avg_bleu4_score))

        info_txt = 'test_dataset_size: {}, avg_bleu4_score:{}.'.format(len(test_dataset), avg_bleu4_score)
        log.info(info_txt, save_to_file=True)

        return avg_bleu4_score

    
    def print_and_log(self, info: str, accelerator: Accelerator=None) -> None:
        '''
        使用accelerator.print, 否则多进程打印会异常
        '''
        if not accelerator:
            print(info)
        else:
            accelerator.print(info)
        self.logger.info(info, std_out=False, save_to_file=True)

if __name__ == '__main__':
    
    # trainer = ChatTrainer()
    train_config = TrainConfig()
    model_config = T5ModelConfig()

    chat_trainer = ChatTrainer(train_config=train_config, model_config=model_config)

    chat_trainer.train()
    # chat_trainer.test(best_epoch=0)