from torch.utils.data import Dataset
from tokenizers import Tokenizer
from fastparquet import ParquetFile
from os.path import dirname, abspath, exists
from os import remove, mkdir
from torch.utils.data import DataLoader
from datasets import load_dataset

ROOT_PATH = abspath(dirname(dirname(__file__)))

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

        # 统计__getitem__调用了多少次，当调用次数大于数据集长度length时，返回空数据
        self.get_cnt = 0
    
    def item_generator(self,) -> tuple:
        '''
        一条数据的生成器，大数据集OOM
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
 
    def __init__(self,  parquet_file: str, tokenizer_file: str, max_len: int=256) -> None:
        '''
        使用huggingface的loaddataset方法加载
        '''

        tokenizer = Tokenizer.from_file(tokenizer_file)
        tokenizer.enable_padding(length=max_len)
        tokenizer.enable_truncation(max_length=max_len)
        self.tokenizer = tokenizer

        self.encode_batch = self.tokenizer.encode_batch
        
        dataset = load_dataset('parquet', data_files=parquet_file, streaming=True) # streaming=True,否则大数据集OOM

        # 这里的batch_size不是训练的batch_size，是传递给precess_batch_func的batch_size
        dataset = dataset.map(self.precess_batch_func, batched=True, batch_size=4096, \
                              remove_columns=['question', 'answer'],  fn_kwargs={'encode_batch': self.encode_batch})
        
        dataset = dataset.with_format(type="torch")

        self.dataset = dataset
    
    def precess_batch_func(self, item: dict, encode_batch: object) -> dict:
        '''
        处理一个批次的文本，转换为id，并返回mask
        '''
        question = encode_batch(item['question'])
        answer = encode_batch(item['answer'])

        q_ids, q_mask = [q.ids for q in question], [q.attention_mask for q in question]
        a_ids, a_mask = [a.ids for a in answer], [a.attention_mask for a in answer]

        return {'q_ids': q_ids, 'q_mask': q_mask, 'a_ids': a_ids, 'a_mask': a_mask}
    
    def __getitem__(self, index: str) -> Dataset:
        '''
        魔术方法，实现dataset['train']、dataset['validation']、dataset['test']
        '''
        return self.dataset[index]

if __name__ == '__main__':
    parquet_file = ROOT_PATH + '/data/my_test_dataset.parquet'
    tokenizer_file = ROOT_PATH + '/model_save/my_tokenizer.250w.20480token.json'

    # example 1：
    # dataset = MyDataset(parquet_file, tokenizer_file)
    # print(len(dataset))
    # dataloader = DataLoader(dataset, batch_size=16)

    # for x, x_mask, y, y_mask in dataloader:
    #     print(x, x_mask, y, y_mask)
    #     break

    dataset = ParquetDataset(parquet_file, tokenizer_file, max_len=16)
    dataloader = DataLoader(dataset['train'], batch_size=16)
    for batch in dataloader:
        x, x_mask, y, y_mask = batch['q_ids'], batch['q_mask'], batch['a_ids'], batch['a_mask']
        print(x.shape, x_mask.shape, y.shape, y_mask.shape)
        break
 
        
    