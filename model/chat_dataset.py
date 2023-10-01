import sys
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from fastparquet import ParquetFile
from os.path import dirname, abspath
from torch.utils.data import DataLoader
from datasets import load_dataset
import datasets

sys.path.append('.')
sys.path.append('..')

from config import PROJECT_ROOT

class MyDataset(Dataset):

    def __init__(self, parquet_file: str, tokenizer_file: str, max_len: int=256) -> None:
        '''
        使用pytorch内置方法加载
        '''
        super().__init__()
        self.parquet_data = ParquetFile(parquet_file)

        # 获取数据集长度
        cnt = 0
        for pf_chunk in self.parquet_data:
            cnt += pf_chunk.info['rows']
        
        self.length = cnt

        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        self.tokenizer.enable_padding(length=max_len)
        self.tokenizer.enable_truncation(max_length=max_len)

        # 统计__getitem__调用了多少次，当调用次数大于数据集长度length时，返回空数据
        self.get_cnt = 0
    
    def item_generator(self,) -> tuple:
        '''
        一条数据的生成器，防止大数据集OOM
        '''
        # 死循环，get_cnt=length时退出
        encode = self.tokenizer.encode
        while True:
            # 当调用次数大于数据集长度length时，返回空数据
            if self.get_cnt > self.length:
                break

            for pf_chunk in self.parquet_data:
                for rows in pf_chunk.iter_row_groups(): #一个pf_chunk大概1万-5万条数据，视写入parquet时的配置定
                    for row in rows.iterrows():
                        
                        # text to ids
                        question = encode(row[1]['question'])
                        answer = encode(row[1]['answer'])

                        self.get_cnt += 1
                        
                        yield question, answer

        
    def __getitem__(self, index):
        '''
        返回一条样本
        '''
        q_encode, a_encode = next(self.item_generator())
       
        return q_encode.ids, q_encode.attention_mask, a_encode.ids, a_encode.attention_mask
    
    def __len__(self) -> int:
        return self.length

class ParquetDataset:
 
    def __init__(self,  parquet_file: str|dict, tokenizer_file: str, buffer_size: int=8192, max_len: int=256, seed: int=23333) -> None:
        '''
        使用huggingface的loaddataset方法加载,
        parquet_file: 单个文件，此时只能使用dataset['train']，
                多个文件请用:parquet_file={'train': 'train.parquet', 'test': 'test.parquet', 'validation': 'validation.parquet'})
                其他用法见：https://huggingface.co/docs/datasets/loading
        '''

        self.len_dict = self.__get_all_parquet_file_size(parquet_file=parquet_file)

        tokenizer = Tokenizer.from_file(tokenizer_file)
        tokenizer.enable_padding(length=max_len)
        tokenizer.enable_truncation(max_length=max_len)
        self.tokenizer = tokenizer

        self.encode_batch = self.tokenizer.encode_batch
        
        dataset = load_dataset('parquet', data_files=parquet_file, streaming=True) # streaming=True,否则大数据集OOM

        # 这里的batch_size不是训练的batch_size，是传递给precess_batch_func的batch_size
        dataset = dataset.map(self.precess_batch_func, batched=True, batch_size=buffer_size, \
                              remove_columns=['question', 'answer'],  fn_kwargs={'encode_batch': self.encode_batch})
        
        dataset = dataset.with_format(type="torch")

        # 只能打乱缓冲区内的数据，不能打乱整个数据集，因此可以将缓存区设置稍微大一些
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)

        self.dataset = dataset
    
    def precess_batch_func(self, item: dict, encode_batch: object) -> dict:
        '''
        处理一个批次的文本，转换为id，并返回mask
        '''
        question = encode_batch(item['question'])
        answer = encode_batch(item['answer'])

        inputs_ids, inputs_mask = [q.ids for q in question], [q.attention_mask for q in question]
        target_ids, target_mask = [a.ids for a in answer], [a.attention_mask for a in answer]

        return {'inputs_ids': inputs_ids, 'inputs_mask': inputs_mask, 'target_ids': target_ids, 'target_mask': target_mask}
    
    def __getitem__(self, index: str) -> datasets.Dataset:
        '''
        魔术方法，实现下标访问，如：dataset['train']、dataset['validation']、dataset['test']
        '''
        return self.dataset[index]
    
    def __get_all_parquet_file_size(self, parquet_file: str|dict) -> dict:
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
        if len(self.len_dict) == 1:
            return self.len_dict['train']
        else:
            raise Exception("this dataset contains many splited dataset, use `get_dataset_size(split_name)` function to get len, e.g: get_dataset_size('train')")
    
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
    # dataset = MyDataset(parquet_file, tokenizer_file)
    # print(len(dataset))
    # dataloader = DataLoader(dataset, batch_size=16)

    # for x, x_mask, y, y_mask in dataloader:
    #     print(x, x_mask, y, y_mask)
    #     break

    # example 2:
    dataset = ParquetDataset(parquet_file, tokenizer_file, max_len=32)
    dataloader = DataLoader(dataset['train'], batch_size=32)
    print(dataset.get_dataset_size('train'))
    step = 0
    for epoch in range(2):
        for batch in dataloader:
            x, x_mask, y, y_mask = batch['inputs_ids'], batch['inputs_mask'], batch['target_ids'], batch['target_mask']
            step += 1
            # print(x.shape, x_mask.shape, y.shape, y_mask.shape)
            # break
            if step % 500 == 0:
                print(step)
    print(step)
 
        
    