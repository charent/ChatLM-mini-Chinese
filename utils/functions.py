from collections import Counter
from typing import Union

import ctypes
import os
import platform

from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import ujson

def get_free_space_of_disk(folder: str='./') -> float:
    '''
    获取指定目录所在磁盘大小，返回单位: GB
    '''
    res_val = 0.0
    if platform.system() == 'Windows':
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(folder), None, None, ctypes.pointer(free_bytes))
        res_val = free_bytes.value 
    else:
        st = os.statvfs(folder)
        res_val = st.f_bavail * st.f_frsize
    
    return res_val / (1024 ** 3)

def my_average(arry_list: list[float]) -> float:
    '''
    自定义均值计算，空数组返回0.0
    '''
    
    if len(arry_list) == 0: return 0.0 
    
    return np.average(arry_list)

def get_path_of_suffix_files(root: str, suffix: str, with_create_time: bool=False) -> list:
    '''
        获取指定目录下下指定后缀的所有文件的绝对路径
    '''
    suffix_files = []
    for root, _, files in os.walk(root):
        for file in files:
            if file.endswith(suffix):
                full_path = '{}/{}'.format(root, file)
                if with_create_time:
                    suffix_files.append( (full_path, os.path.getctime(full_path)) )
                else:
                    suffix_files.append(full_path)
                            
    return suffix_files

def get_bleu4_score(reference: Union[str, list[str]], outputs: Union[str, list[str]], n_gram: int=4) -> float:
    '''
    获取bleu4分数
    '''
    
    weights = np.ones(n_gram) * (1.0 / n_gram)

    outputs_len, reference_len = len(outputs), len(reference)

    if not type(reference) is list:
        reference = list(reference)
    if not type(outputs) is list:
        outputs = list(outputs)

    outputs_counter = extract_Ngram(outputs, n_gram=n_gram)
    reference_counter = extract_Ngram(reference, n_gram=n_gram)

    ngram_counter_clip = outputs_counter & reference_counter

    clip_counter = np.zeros(n_gram)
    output_ngram_counter = np.zeros(n_gram)

    for (key, ngram), cnt in ngram_counter_clip.items():
        clip_counter[ngram - 1] += cnt 
    
    for (key, ngram), cnt in outputs_counter.items():
        output_ngram_counter[ngram - 1] += cnt
    
    # print(clip_counter, output_ngram_counter)
    if np.min(clip_counter) == 0.0:
        return np.array(0.0)

    precision_scores = clip_counter / output_ngram_counter
   
    # bleu
    log_precision_scores = weights * np.log(precision_scores)
    
    # 几何平均形式求平均值然后加权
    geometric_mean = np.exp(np.sum(log_precision_scores))
    brevity_penalty = np.exp(1 - (reference_len / outputs_len))

    # brevity_penalty = 1.0,   bleu = sentence_bleu([reference], outputs)
    # brevity_penalty = 1.0

    bleu = brevity_penalty * geometric_mean

    return bleu


def extract_Ngram(words_list: list[str], n_gram: int) -> tuple:
    '''
    获取一个句子的n_grama
    return：
        ngram_counter： key = ('w1  w2 ... wn', n_gram), value: count of key
    '''
    n = len(words_list)
    ngram_counter = Counter()

    for i in range(1, n_gram + 1):
        for j in range(n - i + 1):
            key = ' '.join(words_list[j: j + i])
            ngram_counter[(key, i)] += 1

    return ngram_counter


def save_model_config(config_dict: dict, file: str) -> None:
    '''
    将模型配置写入到json文件, 输入模型保存的目录及文件名
    '''
    # file = file.replace('\\', '/')
    # file = '{}/model_config.json'.format('/'.join(file.split('/')[0: -1]))
    
    with open(file, 'w', encoding='utf-8') as f:
        ujson.dump(config_dict, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    ref = '抱歉，我不知道ABB代表什么意思'
    out = '我不明白ABB是什么意思'
    b1 = sentence_bleu([list(out)], list(ref),  weights=(0.25, 0.25, 0.25, 0.25))
    print(b1)
    b2 = get_bleu4_score(out, ref)
    print(b2)

    
    candidate_corpus = ['i', 'have', 'a', 'pen', 'on', 'my', 'desk', 'a', 'b', 'c', 'd','f','f']
    reference_corpus = ['there', 'is', 'a', 'pen', 'on', 'my', 'desk', 'a', 'b', 'd', 'd', 'fd']
    
    print('----')
    print(sentence_bleu([reference_corpus], candidate_corpus,  weights=(0.25, 0.25, 0.25, 0.25)))
    print(get_bleu4_score(reference_corpus, candidate_corpus))