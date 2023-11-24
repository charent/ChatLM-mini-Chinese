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
from utils.raw_data_process import write_single_parquet_file

log = Logger('data_process', save2file=True, file_name=PROJECT_ROOT + '/logs/raw_data_process.log')

def generate_bad_response(groups_cnt: int=50000, max_len: int=512) -> None:
    '''生成不是很满意的回答回答
    '''
    print('load model...')
    infer_conf = InferConfig()
    chat_bot = ChatBot(infer_config=infer_conf)

    finetune_file = PROJECT_ROOT + '/data/my_finetune_data_zh.parquet'
    save_rw_file = PROJECT_ROOT + '/data/my_rlhf_dataset.parquet'
    pf = pq.read_table(finetune_file)

    # log.info('process file: {}'.format(finetune_file), save_to_file=True)

    i = 0
    for prompt, response in progress.track(zip(pf['prompt'], pf['response']), total=pf.num_rows):
        reject = chat_bot.chat(str(prompt))
        print(reject)
        break




if __name__ == '__main__':
    generate_bad_response()