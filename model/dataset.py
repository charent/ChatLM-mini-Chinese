from typing import Union

from torch.utils.data import Dataset
from torch import LongTensor, cuda
from transformers import PreTrainedTokenizerFast
from fastparquet import ParquetFile
from torch.utils.data import DataLoader
from datasets import load_dataset
import datasets
import pyarrow.parquet as pq
from numpy import array, int64
from numpy.random import shuffle

# import sys 
# sys.path.extend(['.', '..'])

from config import PROJECT_ROOT

class MyDataset(Dataset):

    def __init__(self, 
                parquet_file: str,
                tokenizer_dir: str,
                keep_in_memory: bool=False,
                max_seq_len: int=512,
                buffer_size: int=40960,
            ) -> None:
        '''
        keep_in_memory: 是否将parquet文件转换为pandas.DataFrame格式存放到内存, 
            False将使用迭代生成器(迭代生成器不支持打乱数据)，减少大数据集内存占用
        '''
        super().__init__()

        if cuda.device_count() >= 2 and not keep_in_memory:
            raise ValueError(f'多GPU时使用MyDataset，参数keep_in_memory必须=True，否则无法进行分布式训练. 当前keep_in_memory={keep_in_memory}')

        self.keep_in_memory = keep_in_memory
        self.max_seq_len = max_seq_len

        # 使用pyarrow.parquet读取，to_pandas、for遍历速度更快
        parquet_table = pq.read_table(parquet_file)

        # 获取数据集长度
        self.length = parquet_table.num_rows

        # 缓冲区大小不能超过数据长度
        self.buffer_size = self.length if buffer_size > self.length else buffer_size

        if keep_in_memory:
            # 转化为pandas放到内存中
            self.data = parquet_table.to_pandas()  
        else:
            self.data = parquet_table

        # 初始化tokenizer
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

        # 在这里初始化generator
        self.sample_generator = self.item_generator()
    
    def item_generator(self,) -> tuple:
        '''
        一条数据的生成器，防止大数据集OOM
        '''
                
        parquet_table = self.data

        # 生成器是死循环，不用退出，训练结束（epoch结束）会停止调用next()
        buffer_list = []
        while True:

            for prompt, response in zip(parquet_table['prompt'], parquet_table['response']):
                
                # 缓存数据不够，添加数据
                if len(buffer_list) < self.buffer_size:
                    buffer_list.append( (prompt.as_py(), response.as_py()) )
                    continue
                
                # 执行到这里，缓存区够了，打乱数据
                shuffle(buffer_list)
                for p, r in buffer_list:
                    # 在这里迭代
                    yield  p, r

                # 迭代完成，清空缓存区
                buffer_list = []
    
    def __getitem__(self, index):
        '''
        返回一条样本
        '''
        if self.keep_in_memory:
            data = self.data
            prompt, response = data.iloc[index].prompt, data.iloc[index].response
        else:
            prompt, response = next(self.sample_generator)

        max_seq_len = self.max_seq_len - 5 # len('[EOS]') = 5
        # add an eos token note that end of resopnse, using in generate.
        return f"{prompt[0: max_seq_len]}[EOS]", f"{response[0: max_seq_len]}[EOS]"

    def collate_fn(self, data: list[list]) -> dict:
        '''
        合并一个批次数据返回
        '''
        tokenizer = self.tokenizer

        prompt = tokenizer([item[0] for item in data], padding=True, return_token_type_ids=False)
        response = tokenizer([item[1] for item in data], padding=True, return_token_type_ids=False)

        input_ids = array(prompt.input_ids, dtype=int64)
        input_mask = array(prompt.attention_mask, dtype=int64)
        target_ids = array(response.input_ids, dtype=int64)

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
                tokenizer_dir: str, 
                keep_in_memory: bool=False,
                cache_dir: str='./.cache',
                buffer_size: int=10240, 
                max_len: int=512, 
                seed: int=23333
            ) -> None:
        '''
        使用huggingface的loaddataset方法加载,
        parquet_file: 单个文件，此时只能使用dataset['train']，
                多个文件请用:parquet_file={'train': 'train.parquet', 'test': 'test.parquet', 'validation': 'validation.parquet'})
                其他用法见：https://huggingface.co/docs/datasets/loading
        keep_in_memory: 是否将parquet文件转换为pandas.DataFrame格式存放到内存
        '''
        self.keep_in_memory = keep_in_memory
        self.len_dict = self.__get_all_parquet_file_size(parquet_file=parquet_file)

        self.max_len = max_len
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

        self.tokenizer = self.tokenizer
        
        streaming = False if keep_in_memory else True 
        # streaming=True,否则大数据集OOM
        dataset = load_dataset('parquet', data_files=parquet_file, cache_dir=cache_dir, streaming=streaming) 

        # 这里的batch_size不是训练的batch_size，是传递给precess_batch_func批处理的batch_size
        dataset = dataset.map(self.precess_batch_func, batched=True, batch_size=buffer_size, \
                            remove_columns=['prompt', 'response'], fn_kwargs={'max_len': max_len})

        dataset = dataset.with_format(type="torch")

        if keep_in_memory:
           dataset = dataset.shuffle(seed=seed, keep_in_memory=keep_in_memory)
        else:
            # 只能打乱缓冲区内的数据，不能打乱整个数据集，因此可以将缓存区设置稍微大一些
            dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)

        self.dataset = dataset
    
    @staticmethod
    def precess_batch_func(item: dict, max_len: int=512) -> dict:
        '''
        添加EOS
        '''
        max_len -= 5 # len('[EOS]') = 5
        for i in range(len(item['prompt'])):
            item['prompt'][i] = f"{item['prompt'][i][0: max_len]}[EOS]"
        for i in range(len(item['response'])):
            item['response'][i] = f"{item['response'][i][0: max_len]}[EOS]"

        return {
            'prompt': item['prompt'],
            'response': item['response'],
        }
    
    def collate_fn(self, data: list[list]) -> dict:
        '''
        合并一个批次数据返回
        '''
        
        tokenizer = self.tokenizer
        prompt = [item['prompt'] for item in data ]
        response = [item['response'] for item in data ]

        # 按批次pad
        prompt_encoded = tokenizer(prompt, padding=True, return_token_type_ids=False)
        response_encoded = tokenizer(response, padding=True, return_token_type_ids=False)

        input_ids = array(prompt_encoded.input_ids, dtype=int64)
        input_mask = array(prompt_encoded.attention_mask, dtype=int64)
        target_ids = array(response_encoded.input_ids, dtype=int64)

        ret = {
            'input_ids': LongTensor(input_ids),
            'input_mask': LongTensor(input_mask),
            'target_ids': LongTensor(target_ids),
        }
        return ret
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
        parquet_data = pq.read_table(file_name)

        return parquet_data.num_rows 
    
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
    
    def get_tokenizer(self, ) -> PreTrainedTokenizerFast:
        return self.tokenizer



if __name__ == '__main__':
    parquet_file = PROJECT_ROOT + '/data/my_valid_dataset.parquet'
    tokenizer_dir = PROJECT_ROOT + '/model_save/tokenizer'

    # example 1：
    dataset = MyDataset(parquet_file, tokenizer_dir, keep_in_memory=False, max_seq_len=128)
    print('\nexample 1, dataset size: ', len(dataset))
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)

    for epoch in range(2):
        print('epoch: {}'.format(epoch))
        for step, batch in enumerate(dataloader):
            x, x_mask, y = batch['input_ids'], batch['input_mask'], batch['target_ids']
            print('step:{}'.format(step), x.shape, x_mask.shape, y.shape)
            if step == 5:
                break

    
    # exit(0)
    # example 2:
    dataset = ParquetDataset(parquet_file, tokenizer_dir, keep_in_memory=True, max_len=32)
    dataloader = DataLoader(dataset['train'], batch_size=32, collate_fn=dataset.collate_fn)
    print('\nexample 2, dataset size: ', dataset.get_dataset_size('train'))

    for epoch in range(2):
        print('epoch: {}'.format(epoch))
        for step, batch in enumerate(dataloader):
            x, x_mask, y = batch['input_ids'], batch['input_mask'], batch['target_ids']
            print('step:{}'.format(step), x.shape, x_mask.shape, y.shape)
            if step == 5:
                break
        
    