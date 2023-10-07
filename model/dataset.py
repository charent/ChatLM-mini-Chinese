from typing import Union
import time

from torch.utils.data import Dataset
from torch import LongTensor
from tokenizers import Tokenizer
from fastparquet import ParquetFile
from torch.utils.data import DataLoader
from datasets import load_dataset
import datasets
import pyarrow.parquet as pq
from numpy import array, int64

import sys 
sys.path.extend(['.', '..'])

from config import PROJECT_ROOT

class MyDataset(Dataset):

    def __init__(self, 
                parquet_file: str,
                tokenizer_file: str,
                keep_in_memory: bool=False,
                max_seq_len: int=256,
            ) -> None:
        '''
        keep_in_memory: 是否将parquet文件转换为pandas.DataFrame格式存放到内存, 
            False将使用迭代生成器(迭代生成器不支持打乱数据)，减少大数据集内存占用
        '''
        super().__init__()

        self.keep_in_memory = keep_in_memory
        self.max_seq_len = max_seq_len

        # 使用pyarrow.parquet读取，to_pandas、for遍历速度更快
        parquet_table = pq.read_table(parquet_file)

        # 获取数据集长度
        self.length = parquet_table.num_rows

        if keep_in_memory:
            # 转化为pandas放到内存中
            self.data = parquet_table.to_pandas()  
        else:
            self.data = parquet_table

        # 初始化tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_file)
        tokenizer.enable_padding(length=max_seq_len)
        tokenizer.enable_truncation(max_length=max_seq_len)
        self.tokenizer = tokenizer
        self.encode = tokenizer.encode
    
    def item_generator(self,) -> tuple:
        '''
        一条数据的生成器，防止大数据集OOM
        '''
        
        parquet_table = self.data

        # 生成器是死循环，不用退出，训练结束（epoch结束）会停止调用next()
        while True:

            for question, answer in zip(parquet_table['question'], parquet_table['answer']):

                yield question.as_py(), answer.as_py()
    
    def __getitem__(self, index):
        '''
        返回一条样本
        '''
        if self.keep_in_memory:
            data = self.data
            question, answer = data.iloc[index][0], data.iloc[index][1]
        else:
            question, answer = next(self.item_generator())

        encode = self.encode
        question_encoded, answer_encoded = encode(question), encode(answer)
       
        return question_encoded.ids, question_encoded.attention_mask, answer_encoded.ids

    @staticmethod
    def collate_fn(data: list[list]) -> dict:
        '''
        合并一个批次数据返回
        '''
        input_ids = array([item[0] for item in data], dtype=int64)
        input_mask = array([item[1] for item in data], dtype=int64)
        target_ids = array([item[2] for item in data], dtype=int64)

        ret = {
            'input_ids': LongTensor(input_ids),
            'input_mask': LongTensor(input_mask),
            'target_ids': LongTensor(target_ids),
        }
        return ret
    
    def __len__(self) -> int:
        return self.length

class ParquetDataset:
 
    def __init__(self,  
                parquet_file: Union[str, dict],
                tokenizer_file: str, 
                keep_in_memory: bool=False,
                cache_dir: str='./cache',
                buffer_size: int=10240, 
                max_len: int=256, 
                seed: int=23333
            ) -> None:
        '''
        parquet迭代速度太慢了！
        parquet迭代速度太慢了！
        parquet迭代速度太慢了！

        使用huggingface的loaddataset方法加载,
        parquet_file: 单个文件，此时只能使用dataset['train']，
                多个文件请用:parquet_file={'train': 'train.parquet', 'test': 'test.parquet', 'validation': 'validation.parquet'})
                其他用法见：https://huggingface.co/docs/datasets/loading
        keep_in_memory: 是否将parquet文件转换为pandas.DataFrame格式存放到内存
        '''
        self.keep_in_memory = keep_in_memory
        self.len_dict = self.__get_all_parquet_file_size(parquet_file=parquet_file)

        tokenizer = Tokenizer.from_file(tokenizer_file)
        tokenizer.enable_padding(length=max_len)
        tokenizer.enable_truncation(max_length=max_len)
        self.tokenizer = tokenizer

        self.encode_batch = self.tokenizer.encode_batch
        
        streaming = False if keep_in_memory else True 
        # streaming=True,否则大数据集OOM
        dataset = load_dataset('parquet', data_files=parquet_file, cache_dir=cache_dir, streaming=streaming) 

        # 这里的batch_size不是训练的batch_size，是传递给precess_batch_func的batch_size
        dataset = dataset.map(self.precess_batch_func, batched=True, batch_size=buffer_size, \
                            remove_columns=['question', 'answer'],  fn_kwargs={'encode_batch': self.encode_batch})
        
        dataset = dataset.with_format(type="torch")

        if keep_in_memory:
           dataset = dataset.shuffle(seed=seed, keep_in_memory=keep_in_memory)
        else:
            # 只能打乱缓冲区内的数据，不能打乱整个数据集，因此可以将缓存区设置稍微大一些
            dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)

        self.dataset = dataset
    
    @staticmethod
    def precess_batch_func(item: dict, encode_batch: object) -> dict:
        '''
        处理一个批次的文本，转换为id，并返回mask
        '''
        question = encode_batch(item['question'])
        answer = encode_batch(item['answer'])

        input_ids, input_mask = [q.ids for q in question], [q.attention_mask for q in question]
        target_ids = [a.ids for a in answer]
        # target_mask = [a.attention_mask for a in answer]

        return {'input_ids': input_ids, 'input_mask': input_mask, 'target_ids': target_ids}
    
    def __getitem__(self, index: str) -> datasets.Dataset:
        '''
        魔术方法，实现下标访问，如：dataset['train']、dataset['validation']、dataset['test']
        '''
        return self.dataset[index]
    
    def __get_all_parquet_file_size(self, parquet_file: Union[str, dict]) -> dict:
        '''
        获取所有parquet file的长度
        '''
        len_dict = dict()
        if type(parquet_file) is str:
            train_len = self.__get_size_of_praquet(parquet_file)
            len_dict['train'] = train_len
        
        if type(parquet_file) is dict:
            for split_type, file in parquet_file.items():
                len_dict[split_type] = self.__get_size_of_praquet(file)
        
        return len_dict
    
    def __get_size_of_praquet(self, file_name: str) -> int:
        '''
        获取一个parquet文件的行数
        '''
        parquet_data = ParquetFile(file_name)

        # 获取数据集长度
        cnt = 0
        for pf_chunk in parquet_data:
            cnt += pf_chunk.info['rows']
        
        return cnt 
    
    def __len__(self) -> int:
        '''
        魔术方法，如果只有一个数据集，返回默认数据集大小
        '''
        if len(self.len_dict) == 1:
            return self.len_dict['train']
        else:
            raise Exception("this dataset contains many splited datasets, use `get_dataset_size(split_name)` function to get length, e.g: get_dataset_size('train')")
    
    def get_dataset_size(self, split_name: str) -> int:
        '''
        获取每个切分数据集的长度
        split_name可取：train、validation、test
        '''
        return self.len_dict[split_name]
    
    def get_tokenizer(self, ) -> Tokenizer:
        return self.tokenizer



if __name__ == '__main__':
    parquet_file = PROJECT_ROOT + '/data/my_valid_dataset.parquet'
    tokenizer_file = PROJECT_ROOT + '/model_save/my_merged_tokenizer.json'

    # example 1：
    dataset = MyDataset(parquet_file, tokenizer_file, keep_in_memory=False)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)

    for epoch in range(3):
        print('epoch: {}'.format(epoch))
        for step, batch in enumerate(dataloader):
            x, x_mask, y = batch['input_ids'], batch['input_mask'], batch['target_ids']
            # print('epoch: {}, step:{},'.format(epoch, step),x.shape, x_mask.shape, y.shape)
            # if step >= 5: break
    
    exit(0)
    # example 2:
    dataset = ParquetDataset(parquet_file, tokenizer_file, keep_in_memory=True, max_len=32)
    dataloader = DataLoader(dataset['train'], batch_size=32,)
    print(dataset.get_dataset_size('train'))
    step = 0
    for epoch in range(2):
        for batch in dataloader:
            x, x_mask, y = batch['input_ids'], batch['input_mask'], batch['target_ids']
            step += 1
            print(x.shape, x_mask.shape, y.shape)
            break
            if step % 500 == 0:
                print(step)
            
    print(step)
 
        
    