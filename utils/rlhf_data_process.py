import sys
sys.path.extend(['.','..'])

import torch
import pandas as pd
import numpy as np
import ujson
from rich import progress
from tokenizers import Tokenizer
from safetensors.torch import load_model

from model.chat_model import TextToTextModel
from logger import Logger
from config import PROJECT_ROOT, InferConfig

from utils.raw_data_process import write_single_parquet_file, punctuation
from utils.functions import json_to_dataclass

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
        

def generate_bad_response(groups_cnt: int=50000, max_len: int=320, batch_size: int=24) -> None:
    '''生成不是很满意的回答回答
    '''
    print('load model...')

    # load config
    infer_config = InferConfig()
    model_config_class = json_to_dataclass(infer_config.model_config_file, 'ModelConfig')
    model_config = model_config_class()

    # load tokenizer and model
    tokenizer = Tokenizer.from_file(infer_config.tokenizer_file)
    tokenizer.enable_padding(length=infer_config.max_seq_len)
    tokenizer.enable_truncation(max_length=infer_config.max_seq_len)
    model = TextToTextModel(config=model_config, decoder_start_token_id=tokenizer.token_to_id('[PAD]'))
    model.load_state_dict(torch.load(infer_config.model_file))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.half()
    model.eval()
    model.to(device)

    finetune_file = PROJECT_ROOT + '/data/alpaca_gpt4_data_zh.json'
    save_rw_json_file = PROJECT_ROOT + '/data/my_rlhf_dataset.json'
    save_rw_parquet_file = PROJECT_ROOT + '/data/my_rlhf_dataset.parquet'

    data = []
    with open(finetune_file, 'r', encoding='utf-8') as f:
        data = ujson.load(f)
    
    log.info('length of {} is {}'.format(save_rw_json_file, len(data)), save_to_file=True)

    model_outs = []
    batch_prompt = []
    process_item = []
    for i, item in progress.track(enumerate(data), total=len(data)):
        # 模型生成的答案为拒绝答案
        # reject = chat_bot.chat(item['prompt'])
        # item['reject'] = reject
        batch_prompt.append(item['prompt'])
        process_item.append(item)
        
        if i % 500 == 0: print('process {} items.'.format(i))

        if len(batch_prompt) >= batch_size or i == len(data) - 1:

            token_encodes = tokenizer.encode_batch(batch_prompt)
            torch.cuda.empty_cache()

            with torch.no_grad():
                input_ids = torch.LongTensor([item.ids for item in token_encodes]).to(device)
                atten_mask = torch.LongTensor([item.attention_mask for item in token_encodes]).to(device)

                outputs = model.generate(input_ids=input_ids, attention_mask=atten_mask, max_seq_len=infer_config.max_seq_len).cpu().numpy()

            outputs = tokenizer.decode_batch(outputs,  skip_special_tokens=True)

            # 删除decode出来字符间的空格
            outputs = [sentance.replace(' ', '') for sentance in outputs]
            model_outs.extend(outputs)
      
            batch_prompt = []
          
        if len(model_outs) % 1000 == 0:
            for i in range(len(model_outs)):
                process_item[i]['reject'] = model_outs[i]
            try:
                with open(PROJECT_ROOT + '/data/outs.ckp.json', 'w', encoding='utf-8') as f:
                    ujson.dump(process_item, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(e)
    
    for i in range(len(model_outs)):
        process_item[i]['reject'] = model_outs[i]

    with open(save_rw_json_file, 'w', encoding='utf-8') as f:
        ujson.dump(process_item, f, indent=4, ensure_ascii=False)
    
    df = pd.DataFrame(data)
    write_single_parquet_file(save_rw_parquet_file, df)


def split_train_eval_dataset() -> None:
    '''划分数据集
    '''
    rw_json_file = PROJECT_ROOT + '/data/my_rlhf_dataset.json'
    train_file = PROJECT_ROOT + '/data/dpo_train.json'
    eval_file = PROJECT_ROOT + '/data/dpo_eval.json'

    data = []

    with open(rw_json_file, 'r', encoding='utf-8') as f:
        data = ujson.load(f)
    
    np.random.shuffle(data)
    split_idx = int(len(data) * 0.95)

    train_data = data[0: split_idx]
    eval_data = data[split_idx: ]

    log.info('train size: {}, eval size:{}'.format(len(train_data), len(eval_data)), save_to_file=True)

    with open(train_file, 'w', encoding='utf-8') as f:
        ujson.dump(train_data, f, indent=4, ensure_ascii=False)

    with open(eval_file, 'w', encoding='utf-8') as f:
        ujson.dump(eval_data, f, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    # 1. 处理chosen文本
    # process_rlhf_chosen_data()

    # 2. 生成rejected文本
    # generate_bad_response()

    # 3. split train and eval dataset
    split_train_eval_dataset()