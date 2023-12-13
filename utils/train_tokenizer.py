from os.path import dirname, abspath, exists
from os import remove, mkdir
import sentencepiece as spm
from fastparquet import ParquetFile
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from rich import progress
import ujson
from jieba_fast import lcut

from multiprocessing import RLock, Pool
from multiprocessing.managers import BaseManager

import pandas as pd
import os 
import time
from collections import defaultdict

from config import PROJECT_ROOT

class MyManager(BaseManager):
    '''
    nothing to be done 
    '''
    pass 

'''
尝试使用 huggingface tokenizers 库进行训练，但是OOM了，GitHub的issue回复也没有实质性建议
故使用sentencepiece训练

'''

def train_my_huggingface_tokenizer() -> None:
    '''
    训练tokenizer with huggingface
    '''

    pf = ParquetFile(PROJECT_ROOT + '/data/my_dataset.parquet')
    tokenizer_save_path = PROJECT_ROOT + '/model_save'
    if not exists(tokenizer_save_path): mkdir(tokenizer_save_path)

    def get_training_corpus():
        for pf_chunk in pf:
            for rows in pf_chunk.iter_row_groups():
                yield rows['prompt'] + '[SEP]' + rows['response'] + '[SEP]'

    model = BPE(unk_token="[UNK]")
    tokenizer = Tokenizer(model)
    # tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(vocab_size=40960, min_frequency=10000, show_progress=True, \
                         special_tokens=["[PAD]", "[CLS]","[SEP]", "[MASK]", "[UNK]"])
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    tokenizer.save(tokenizer_save_path)
   
def train_my_huggingface_wiki_tokenizer(max_train_line: int=1000000) -> None:
    '''
    训练tokenizer with huggingface
    '''

    cropus_file = PROJECT_ROOT + '/data/raw_data/wiki.simple.txt'
    tokenizer_save_path = PROJECT_ROOT + '/model_save/hf_bpe_tokenizer'

    # if not exists(tokenizer_save_path): mkdir(tokenizer_save_path)

    def get_training_corpus(buffer_size: int=10000) -> list:
        line_cnt = 0
        with open(cropus_file, 'r', encoding='utf-8') as f_read:
            cur_rows = []
            for line in f_read:
                if len(line) < 32: continue
                cur_rows.append(line)
                line_cnt += 1
                if len(cur_rows) >= buffer_size:
                    yield cur_rows
                    cur_rows = []
                
                if line_cnt >= max_train_line:
                    break

            if len(cur_rows) > 0:
                yield cur_rows

    model = BPE(unk_token="[UNK]")
    tokenizer = Tokenizer(model)
    # tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(vocab_size=40960, min_frequency=1000, show_progress=True, \
                         special_tokens=["[PAD]", "[CLS]","[SEP]", "[MASK]", "[UNK]"])
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    tokenizer.save(tokenizer_save_path)

def train_my_BPE_tokenizer() -> None:
    '''
    使用sentencepiece训练BPE，缺点只能加载300万行，16G内存会OOM
    '''
    txt_corpus_file = PROJECT_ROOT + '/data/my_corpus.txt'
    special_tokens = ["[PAD]", "[CLS]","[SEP]", "[MASK]", "[UNK]"]

    spm.SentencePieceTrainer.train(
        input=txt_corpus_file, 
        model_prefix='my_tokenizer', 
        vocab_size=40960, 
        user_defined_symbols=special_tokens,
        max_sentence_length=256,
        shuffle_input_sentence=True,
        # character_coverage=1.0,
        model_type='bpe',
    )


def get_corpus_dict() -> None:
    '''
    获取语料的字典
    '''
    parquet_file = PROJECT_ROOT + '/data/my_dataset.parquet'
    save_file = PROJECT_ROOT + '/data/my_dict.json'

    if exists(save_file): remove(save_file)

    source_pf = ParquetFile(parquet_file)
    word_freq = dict()
    single_word_freq = dict()

    def add_to_dict(w_dict: dict, word: str, is_whole_word: bool=True) -> None:
        '''
        将word添加到字典中
        '''
        if is_whole_word:
            w_dict[word] = w_dict.get(word, 0) + 1
        else:
            for ch in word:
                w_dict[ch] = w_dict.get(ch, 0) + 1

    def add_sentence_to_dict(sentence: list) -> None:
        for word in sentence:
            add_to_dict(word_freq, word, is_whole_word=True)
            add_to_dict(single_word_freq, word, is_whole_word=False)

    for pf_chunk in progress.track(source_pf):
        for rows in pf_chunk.iter_row_groups():
            for prompt, response in zip(rows['prompt'], rows['response']):
                prompt, response = lcut(prompt), lcut(response)

                add_sentence_to_dict(prompt)
                add_sentence_to_dict(response)

    
    with open(save_file, 'w', encoding='utf-8') as f:
        ujson.dump({'word_freq': word_freq, 'single_word_freq': single_word_freq}, f,  indent=4, ensure_ascii=False)


def df_process_function(rows: pd.DataFrame, word_freq_dict: dict, single_word_dict:dict, \
                        word_freq_dict_lock: RLock, single_word_dict_lock: RLock)-> None:
    i = 0
    s = time.time()
    for row in rows.iterrows():
        # s = time.time()
        prompt = row[1]['prompt']
        response = row[1]['response']
        prompt_cut, response_cut = lcut(prompt), lcut(response)
        # print('pid {}, q len{}, a len {}, cut use {}s'.format(os.getpid(), len(prompt), len(response), time.time() - s))

        # s = time.time()
        # add_sentence_to_dict(prompt, word_freq_dict, single_word_dict, word_freq_dict_lock, single_word_dict_lock)
        # add_sentence_to_dict(response, word_freq_dict, single_word_dict, word_freq_dict_lock, single_word_dict_lock)

        # with word_freq_dict_lock:
        word_freq_dict_lock.acquire()

        for word in prompt_cut:
            word_freq_dict[word] += 1
        for word in response_cut:
            word_freq_dict[word] += 1

        word_freq_dict_lock.release()

        # --------------------------------------------------------

        # with single_word_dict_lock:
        single_word_dict_lock.acquire()

        for ch in prompt:
            single_word_dict[ch] += 1
        for ch in response:
            single_word_dict[ch] += 1

        single_word_dict_lock.release()

        # print('pid {}, add use {}s'.format(os.getpid(), time.time() - s))
        i += 1
        if i % 100 == 0:
            print('pid {}, {} row, add use {}s'.format(os.getpid(), i, time.time() - s))
            

def add_to_dict(w_dict: dict, lock: RLock,  word: str, is_whole_word: bool=True) -> None:
    '''
    将word添加到字典中
    '''
    s = time.time()
    with lock:
        print('pid {}, wait use {}s'.format(os.getpid(), time.time() - s))

        s = time.time()
        if is_whole_word:
                w_dict[word] = w_dict.get(word, 0) + 1
        else:
            for ch in word:
                w_dict[ch] = w_dict.get(ch, 0) + 1
        print('pid {}, add for loop use {}s'.format(os.getpid(), time.time() - s))

def add_sentence_to_dict(sentence: list,  word_freq_dict, single_word_dict, word_freq_dict_lock, single_word_dict_lock) -> None:
    for word in sentence:
        add_to_dict(word_freq_dict, word_freq_dict_lock, word, is_whole_word=True)
        add_to_dict(single_word_dict, single_word_dict_lock, word, is_whole_word=False)

def run_process(processes_pool: Pool, cur_rows: list, word_freq_dict, single_word_dict, word_freq_dict_lock, single_word_dict_lock) -> None:
    async_res_list = []
    for i in range(len(cur_rows)):
        async_result = processes_pool.apply_async(df_process_function, 
                                                    args=(
                                                            cur_rows[i], 
                                                            word_freq_dict, 
                                                            single_word_dict, 
                                                            word_freq_dict_lock, 
                                                            single_word_dict_lock
                                                        ))
        
        async_res_list.append(async_result)

    # 等等所有进程完成后再处理下一批次数据   
    for async_res in async_res_list:
        async_res.wait()

def get_cropus_dict_multi_process()-> None:

    # 注册共享对象，exposed为暴露的可调用函数函数，__getitem__、__setitem__魔术方法可进行下标访问

    MyManager.register('word_freq_dict', defaultdict, exposed=('__getitem__', '__setitem__', 'get', 'items'))
    MyManager.register('single_word_dict', defaultdict, exposed=('__getitem__', '__setitem__', 'get', 'items'))
    MyManager.register('word_freq_dict_lock', RLock, exposed=('acquire', 'release', '__repr__'))
    MyManager.register('single_word_dict_lock', RLock, exposed=('acquire', 'release', '__repr__'))

    with MyManager() as manager:
        
        cur_rows = []
       
        # 多进程数量
        n = 4
        processes_pool = Pool(n)

        word_freq_dict = manager.word_freq_dict(int)
        single_word_dict = manager.single_word_dict(int)

        word_freq_dict_lock = manager.word_freq_dict_lock()
        single_word_dict_lock = manager.single_word_dict_lock()

        source_pf = ParquetFile(PROJECT_ROOT + '/data/my_dataset.parquet')
        
        for pf_chunk in progress.track(source_pf):
            for rows in pf_chunk.iter_row_groups():
                cur_rows.append(rows)
               
                if len(cur_rows) == n:
                    run_process(processes_pool, cur_rows, word_freq_dict, single_word_dict, word_freq_dict_lock, single_word_dict_lock)
                    cur_rows = []

        if len(cur_rows) > 0:
            run_process(processes_pool, cur_rows, word_freq_dict, single_word_dict, word_freq_dict_lock, single_word_dict_lock)
            
        processes_pool.close()

        word_freq = dict()
        for k,v in word_freq_dict.items(): word_freq[k] = v

        single_word_freq = dict()
        for k,v in single_word_dict.items(): single_word_freq[k] = v

        with open(PROJECT_ROOT + '/model_save/my_vocab.json', 'w', encoding='utf-8') as f:
            ujson.dump({'word_freq': word_freq, 'single_word_freq': single_word_freq}, f,  indent=4, ensure_ascii=False)

def merge_cropus_dict(word_min_freq: int=2500, char_min_freq: int=1500) -> None:
    '''
    合并字典，剔除词频过低的字词
    '''
    raw_dict_file = PROJECT_ROOT + '/model_save/my_dict.json'
    
    raw_dict = None
    with open(raw_dict_file, 'r', encoding='utf-8') as f:
        raw_dict = ujson.load(f)
    
    merged_dict = defaultdict(int)
    for k, v in progress.track(raw_dict['word_freq'].items()):
        if v >= word_min_freq:
            merged_dict[k] = v 
    
    for k, v in progress.track(raw_dict['single_word_freq'].items()):
        if v >= char_min_freq and k not in merged_dict:
            merged_dict[k] = v 

    # special_tokens = ["[PAD]", "[CLS]","[SEP]", "[MASK]", "[UNK]"]
    # for st in special_tokens:
    #     merged_dict[st] = word_min_freq * 10

    print('merged dict len: {}'.format(len(merged_dict)))

    with open(PROJECT_ROOT + '/model_save/my_vocab_merged.dict.json', 'w', encoding='utf-8') as f:
        ujson.dump(merged_dict, f,  indent=4, ensure_ascii=False)


def change_cropus_dict_to_tokenize() -> None:
    '''
    将结巴分词对所有数据集进行分词后统计词频的数据转换未huggingface的tokenizer
    为什么这样做？
        因为各个领域的语料数量不平衡，使用spm or tokenizer.model训练出来的切词效果较差，不具有普适性，如、“贷款，”、“单曲《”，“、赵”，
    '''
    cropus_dict_file = PROJECT_ROOT + '/model_save/my_vocab_merged.dict.json'
    cropus_dict = dict()
    with open(cropus_dict_file, 'r', encoding='utf-8') as f:
        cropus_dict = ujson.load(f)

    print('cropus_dict size: {}'.format(len(cropus_dict)))

    special_tokens = ["[PAD]", "[CLS]","[SEP]", "[BOS]", "[EOS]", "[MASK]", "[UNK]"]

    # 给每个字编号
    words_dict = dict()
    idx = 0
    for token in special_tokens:
        words_dict[token] = idx 
        idx += 1
    
    for word in cropus_dict.keys():
        if word not in words_dict:
            words_dict[word] = idx 
            idx += 1
    
    # add space
    words_dict[' '] = idx

    # 构造merge数组
    words_merge_list = []
    for word in words_dict.keys():
        n = len(word)
        if n >= 2:
            # a, b切分12345示例： 1 2345,  12 345,   123 45,   1234 5
            for i in range(1, n):
                a, b = ''.join(word[0: i]), ''.join(word[i: ])

                if a in words_dict and b in words_dict:
                    words_merge_list.append((a, b))

    print('total word vcoab size: {}'.format(len(words_dict)))

    # 转换为huggingface的tokenizer
    model = BPE(vocab=words_dict, merges=words_merge_list, unk_token='[UNK]')
    tokenizer = Tokenizer(model)
    tokenizer.add_special_tokens(special_tokens)
        
    tokenizer.save(PROJECT_ROOT + '/model_save/my_merged_tokenizer.json')

def trained_tokenizer_to_PreTrainedTokenizerFast():
    '''
    将Tokenizer转换为 PreTrainedTokenizerFast
    '''
    from transformers import PreTrainedTokenizerFast

    tokenizer_obj = Tokenizer.from_pretrained(PROJECT_ROOT + '/model_save/my_merged_tokenizer.json')
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
    tokenizer.pad_token = '[PAD]'
    tokenizer.pad_token_id = tokenizer_obj.token_to_id('[PAD]')
    tokenizer.unk_token = '[UNK]'
    tokenizer.unk_token_id = tokenizer_obj.token_to_id('[UNK]')
    tokenizer.eos_token = '[EOS]'
    tokenizer.eos_token_id = tokenizer_obj.token_to_id('[EOS]')

    tokenizer.save_pretrained(PROJECT_ROOT + '/model_save/tokenizer')


if __name__ == '__main__':
    # train_my_huggingface_tokenizer()

    # train_my_huggingface_wiki_tokenizer()

    # train_my_BPE_tokenizer()
    # get_corpus_dict()
    # get_cropus_dict_multi_process()

    # merge_cropus_dict() 
    # change_cropus_dict_to_tokenize()   
    # trained_tokenizer_to_PreTrainedTokenizerFast()

    pass


