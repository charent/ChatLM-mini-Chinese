import sys
sys.path.extend(['.','..'])

import ujson
import re
from os.path import dirname, abspath, exists, isdir
from os import remove, mkdir, walk
import time
from collections import defaultdict

from matplotlib import pyplot as plt
import codecs, csv
import pandas as pd 
import numpy as np
from rich import progress
from rich.table import Table
from rich.console import Console
from fastparquet import ParquetFile, write
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from model.infer import ChatBot
from logger import Logger
from config import PROJECT_ROOT, InferConfig
from utils.functions import get_path_of_suffix_files
from utils.raw_data_process import write_single_parquet_file, punctuation

log = Logger('data_process', save2file=True, file_name=PROJECT_ROOT + '/logs/raw_data_process.log')

# 结束标点符号
END_PUN = set(".。!！）)》>}】?？\"”\n")

def fixed_response(item: str) -> str:
    '''
    修复被截断的回答，从末尾往回找第一个结束标点
    '''
    if item[-1] in END_PUN: return item

    n = len(item)
    i = n - 1
    while i > 0 and item[i] not in END_PUN:
        i -= 1

    return ''.join(item[0: i + 1])


def process_rlhf_chosen_data(max_len: int=320) -> None:
    ''''
    处理RM高质量回答部分
    数据集：https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh
    '''

    read_file = PROJECT_ROOT + '/data/raw_data/alpaca_gpt4_data_zh.json'
    save_file = PROJECT_ROOT + '/data/alpaca_gpt4_data_zh.json'
    
    my_data = []

    with open(read_file, 'r', encoding='utf-8') as f:
        data = ujson.load(f)
        print('length of {} is {}'.format(read_file, len(data)))
        for item in progress.track(data):
            prompt = item['instruction']
            inputs = item['input']

            # 如果resopnse不是以标点符号结尾，往回找第一个结束符号
            response = fixed_response(item['output'])

            if len(response) > max_len: continue  # 超长的不要

            if len(inputs) > 0:
                if prompt[-1] not in END_PUN:
                    prompt = '{}\n{}'.format(prompt, inputs)
            
            if  len(prompt) > max_len: continue

            my_data.append(
                {
                    'prompt': prompt,
                    'chosen': response
                }
            )

    print('length of {} is {}'.format(save_file, len(my_data)))

    with open(save_file, 'w', encoding='utf-8') as f:
        ujson.dump(my_data, f, indent=4, ensure_ascii=False)
        

def generate_bad_response(groups_cnt: int=50000, max_len: int=320) -> None:
    '''生成不是很满意的回答回答
    '''
    print('load model...')
    infer_conf = InferConfig()
    chat_bot = ChatBot(infer_config=infer_conf)

    finetune_file = PROJECT_ROOT + '/data/alpaca_gpt4_data_zh.json'
    save_rw_json_file = PROJECT_ROOT + '/data/my_rlhf_dataset.json'
    save_rw_parquet_file = PROJECT_ROOT + '/data/my_rlhf_dataset.parquet'

    data = []
    with open(finetune_file, 'r', encoding='utf-8') as f:
        data = ujson.load(f)

    for item in progress.track(data):
        # 模型生成的答案为拒绝答案
        reject = chat_bot.chat(item['prompt'])
        item['reject'] = reject

    with open(save_rw_json_file, 'w', encoding='utf-8') as f:
        ujson.dump(data, f, indent=4, ensure_ascii=False)
    
    df = pd.DataFrame(save_rw_json_file)
    write_single_parquet_file(save_rw_parquet_file, df)



if __name__ == '__main__':
    # process_rlhf_chosen_data()
    generate_bad_response()