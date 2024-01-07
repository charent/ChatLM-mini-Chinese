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
from opencc import OpenCC

import sys
sys.path.extend(['.','..'])

from logger import Logger
from config import PROJECT_ROOT
from utils.functions import get_path_of_suffix_files, DropDatasetDuplicate

log = Logger('data_process', save2file=True, file_name=PROJECT_ROOT + '/logs/raw_data_process.log')

punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：\n")
en_punctuation = ",().!;:"
zh_punctuation = "，（）。！；："

def delete_file(file: str)-> bool:
    '''
    询问删除文件
    '''
    if exists(file):
        ans = input('delete file: {} ? Yes (y) or No (n)'.format(file))
        ans = ans.lower()
        if ans in ('yes', 'y'):
            remove(file)
            print('deleted.')
            return True
    return False

def remove_duplicate_punctuation(sentence: str) -> str:
    '''
    删除句子中重复的标点符号、重复的空格，同时将换行变为特殊字符'\n'
    '''
    # 将空格（全角空格）替换为逗号, 可能会有重复的空客，下面删除重复标点会删除
    sentence = re.sub(' |　', '，', sentence) 

    ans = ''
    n = len(sentence)
    p = 0
    while p < n:
        ans += sentence[p]

        while p + 1 < n and sentence[p] in punctuation and sentence[p + 1] in punctuation:
            p += 1
        p += 1

    return ans

def convert_en_punctuation_to_zh_punct(sentence: str) -> str:
    '''
    将句子中的英文标点替换文中文标点
    '''
    n = len(zh_punctuation)
    for i in range(n):
        sentence = sentence.replace(en_punctuation[i], zh_punctuation[i])
    return sentence

def get_sentences_dice_similarity(st_a: str, st_b: str) -> float:
    '''
    获取两个句子的Dice相似度（Dice similarity）
    s(a, b) =  2 * len( set(a) & set(b) ) / (len(set(a)) + len(set(b)))
    '''
    set_a, set_b = set(st_a), set(st_b)
    total_len  = len(set_a) + len(set_b)
    
    if total_len == 0: return 0.0

    inter_set =  set_a & set_b
    
    return ( 2 * len(inter_set)) / total_len

def write_single_parquet_file(file_name: str, data_frame: pd.DataFrame) -> None:
    '''
    将dataframe写到单独的parquet file中
    '''
    append = False
    if exists(file_name):
        append = True 

    write(file_name, data_frame, compression='GZIP',append=append)


def read_and_write_template(read_file: str, write_to_file: str, call_back: object, group_cnt: int=10000) -> None:
    '''
    处理数据读写模板，需要提供一个回调函数call_back，
    read_file: 原始数据文件
    write_to_file：处理后的要保存数据文件
    call_back：函数输入一个字符串，输出一个处理后的字典dict，如果输入的字符串为无效数据，请返回None
    group_cnt: parquet file分割行数
    如：
    >>> def call_back(inputs: str) -> dict:
    >>>     if check(inputs) not valid:
    >>>         return None
    ...    
    ...    do something for inputs
    ...
    >>>     my_dict = {
    >>>             'prompt': inputs['p'],
    >>>             'response': inputs['a1'] + inputs['a2'],
    >>>             ...
    >>>         }
    >>>     return my_dict
    '''

    log.info('process file:{}'.format(read_file), save_to_file=True)
    start = time.time()
    
    raw_line_cnt = 0
    keep_line_cnt = 0
    
    with progress.open(read_file, 'r', encoding='utf-8') as f_read:
        cur_rows = []
        append = cur_rows.append
        for line in f_read:
            try:
                raw_line_cnt += 1

                write_dict = call_back(line)

                if write_dict is None: continue

                keep_line_cnt += 1
                append(write_dict)
                # ujson.dump(write_obj, f_write, indent=4, ensure_ascii=False)
                # ujson.dump(write_obj, f_write,  ensure_ascii=False,)
                # f_write.write('\n')

                if len(cur_rows) >= group_cnt:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(write_to_file, df)
                    cur_rows = []
                    append = cur_rows.append

            except Exception as e:
                # log.error('处理文件异常：{}, content:{}'.format(str(e), line))
                print(line)
                raise e
        
        # end for
        # 处理末尾部分
        if len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(write_to_file, df)
            cur_rows = []
    
    end = time.time()

    log.info('原始文件:{}，共{}行，处理后剩余{}行，保存到文件：{}。耗时：{:.6}s'\
                .format(read_file, raw_line_cnt, keep_line_cnt, write_to_file, end - start), save_to_file=True)



#=====================================数据集处理=================================

def process_web_text(keep_start: int=5, response_less_word: int=10) -> None:
    '''
    处理425万社区问答webtext2019zh知识类数据集
    keep_start: 只保留点赞数大于keep_start的问答
    response_less_word: 答案至少要有response_less_word个字
    '''
    file_names = [
        '/data/raw_data/web_text_zh_test.json',
        '/data/raw_data/web_text_zh_train.json',
        '/data/raw_data/web_text_zh_valid.json',
    ]

    save_file_name = PROJECT_ROOT + '/data/my_data/my_web_text_zh.parquet'

    # 后续append写入，存在文件先删除
    if exists(save_file_name): 
        assert delete_file(save_file_name)

    def process_function(line: str) -> dict:
        item = ujson.loads(line)

        if item['star'] < keep_start or len(item['content']) < response_less_word: 
            return None

        # 数据清洗
        # 去除重复的标点符号
        prompt = remove_duplicate_punctuation(item['title'])
        response = remove_duplicate_punctuation(item['content'])
        write_dict = {
            "prompt": prompt,
            "response": response,
        }
        return write_dict

    for file_name in file_names:
        read_file = PROJECT_ROOT + file_name

        read_and_write_template(read_file, save_file_name, process_function)


def process_bake_qa(response_less_word: int=15) -> None:
    '''
    处理147万百度知道知识类数据集

    '''
    file_names = [
        '/data/raw_data/baike_qa_train.json',
        '/data/raw_data/baike_qa_valid.json',
    ]

    save_file_name = PROJECT_ROOT + '/data/my_data/my_baike_qa.parquet'
    # 后续append写入，存在文件先删除
    if exists(save_file_name): 
        assert delete_file(save_file_name)

    def process_function(line: str) -> dict:
        item = ujson.loads(line)

        if len(item['answer']) < response_less_word: 
            return None

        # 数据清洗
        prompt = ''
        if get_sentences_dice_similarity(item['title'], item['desc']) >= 0.90:
            # title 和desc 相似度过高，只用title作为问题
            prompt = item['title']
        else:
            # title 和desc拼接形成问题
            prompt = "{}{}".format(item['title'], item['desc'])

        # 删除\r
        prompt = prompt.replace('\r','') 

        # 删除重复的标点符号
        prompt = remove_duplicate_punctuation(prompt)

        # 去除重复的标点符号
        response = item['answer'].replace('\r','')
        response = remove_duplicate_punctuation(response)

        # 剔除问题和答案过短的数据
        if len(prompt) < 3 or len(response) < response_less_word:
            return None
        
        write_dict = {
                "prompt": prompt,
                "response": response,
            }

        return write_dict

    for file_name in file_names:
        read_file = PROJECT_ROOT + file_name
        
        read_and_write_template(read_file, save_file_name, process_function)

  
def repair_line_error_csv_file(raw_csv_file: str, save_suffix: str, read_encoding: str='utf-8', ) -> None:
    '''
        修复csv文件，将文件中换行符替换为\n，字段中的英文字符替换为中文字符
    '''
    
    with codecs.open(raw_csv_file, 'r', encoding=read_encoding, errors='ignore') as f:
        reader = csv.reader(f)
        new_lines = []

        for line in reader:
            for i in range(len(line)):
                line[i] = line[i].replace('\n', '\\n') # 处理异常的换行符
                line[i] = line[i].replace(',', '，') # 英文逗号换为中文逗号
            new_lines.append(line)

        with open(raw_csv_file[: -4] + save_suffix, 'w', encoding='utf-8', newline="") as f:
            writer = csv.writer(f)
            writer.writerows(new_lines)

def process_chinese_medical_datasets(response_less_word: int=15) -> None:
    '''
    处理中国医药领域问答数据集
    '''
    raw_dataset_dir = PROJECT_ROOT + '/data/raw_data/chinese_medical_dialogue_datasets'
    
    raw_data_files = get_path_of_suffix_files(raw_dataset_dir, '.csv')

    # 如果没有修复的文件，则修复csv文件换行异常
    suffix = '.repaired.csv'
    need_to_repair_files = [
        file_name for file_name in raw_data_files \
            if not file_name.endswith(suffix) and file_name[0: -4] + suffix not in raw_data_files
    ]
 
    # 修复异常换行的文件
    for file_name in need_to_repair_files:
        repair_line_error_csv_file(file_name, suffix, read_encoding='gb2312')

    # 重新获取原始文件（即修复后的文件）
    raw_data_files = get_path_of_suffix_files(raw_dataset_dir, suffix)

    # 获取要保存的文件名
    save_file = PROJECT_ROOT + '/data/my_data/my_chinese_medical_dialogue.parquet'
    # for file_name in raw_data_files:
    #     file_name = file_name.split('/')[-1][0: -(len(suffix))] + '.parquet'
    #     file_name = PROJECT_ROOT  + '/data/my_data/' + file_name
    #     save_files.append(file_name)

    # 后续append写入，存在文件先删除
    if exists(save_file): 
        assert delete_file(save_file)
    
    def process_function(line: str) -> dict:
        # department,title,ask,answer
        item = line.split(',') # csv文件逗号分割
        if len(item) < 4:
            print(item)
            return None

        if len(item[3]) < response_less_word: 
            return None

        # 数据清洗
        prompt = ''
        if get_sentences_dice_similarity(item[1], item[2]) >= 0.90:
            # title 和ask 相似度过高，只用ask作为问题
            prompt = item[2]
        else:
            # title 和 ask 拼接形成问题
            prompt = "{}{}".format(item[1], item[2])

        # 删除\r
        prompt = prompt.replace('\r','') 

        # 删除重复的标点符号
        prompt = remove_duplicate_punctuation(prompt)

        # 去除重复的标点符号
        response = ''.join(item[3: ]).replace('\r','')
        response = remove_duplicate_punctuation(response)

        # 剔除问题和答案过短的数据
        if len(prompt) < 3 or len(response) < response_less_word:
            return None
        
        write_dict = {
                "prompt": prompt,
                "response": response,
            }

        return write_dict

    for i, file_name in enumerate(raw_data_files):
        read_file = file_name        

        read_and_write_template(read_file, save_file, process_function)


def process_finace_dataset(prompt_less_word: int=10, response_less_word: int=15) -> None:
    '''
    处理金融问答数据集
    '''
    finace_data_file = PROJECT_ROOT + '/data/raw_data/financezhidao_filter.csv'
    
    suffix = '.repaired.csv'
    if not exists(finace_data_file[0: -4] + suffix):
        repair_line_error_csv_file(finace_data_file, save_suffix=suffix, read_encoding='utf-8')

    
    def process_function(line: str) -> dict:
        # title,prompt,reply,is_best
        item = line.split(',') # csv文件逗号分割
        if len(item) < 4:
            print(item)
            return None

        if len(item[0]) + len(item[1]) < prompt_less_word or len(item[2]) < response_less_word: 
            return None

        # 数据清洗
        prompt = ''
        if get_sentences_dice_similarity(item[0], item[1]) >= 0.90:
            # title 和prompt 相似度过高，只用最长的作为问题
            prompt = item[0] if len(item[0]) > len(item[0]) else item[1]
        else:
            # title 和 ask 拼接形成问题
            prompt = "{}{}".format(item[0], item[1])

        # 删除\r
        prompt = prompt.replace('\r','') 

        # 删除重复的标点符号
        prompt = remove_duplicate_punctuation(prompt)

        # 去除重复的标点符号
        response = ''.join(item[2]).replace('\r','')
        response = remove_duplicate_punctuation(response)

        # 剔除问题和答案过短的数据
        if len(prompt) < prompt_less_word or len(response) < response_less_word:
            return None
        
        write_obj = {
                "prompt": prompt,
                "response": response,
            }

        return write_obj

  
    read_file = finace_data_file[0: -4] + suffix
    write_file = PROJECT_ROOT + '/data/my_data/' + read_file.split('/')[-1][0: -(len(suffix))] + '.parquet'

    # 后续append写入，存在文件先删除
    if exists(write_file): 
        assert delete_file(write_file)

    read_and_write_template(read_file, write_file, process_function)


def process_zhihu_kol_dataset(prompt_less_word: int=4, response_less_word: int=10, group_cnt: int=10000) -> None:
    '''
    处理知乎数据集
    
    '''
    raw_zhihu_data_path = abspath(dirname(dirname(__file__))) + '/data/raw_data/zhihu-kol'
    file_names = []
    suffix = '.parquet'
    for root, _, files in walk(raw_zhihu_data_path):
        for file in files:
            if file.endswith(suffix):
                file_names.append(root + '/' + file)
    
    
    def process_function(sentence: str) -> str:
        '''
        针对一个句子的数据清洗
        '''
        # 删除\r
        sentence = sentence.replace('\r','') 

        # 删除重复的标点符号
        sentence = remove_duplicate_punctuation(sentence)

        return sentence

    # row keys :['INSTRUCTION', 'RESPONSE', 'SOURCE', 'METADATA']
    save_file = PROJECT_ROOT + '/data/my_data/zhihu_kol.parquet'
    
    # 后续append写入，存在文件先删除
    if exists(save_file): 
        assert delete_file(save_file)

    all_cnt, keep_cnt = 0, 0
    cur_rows = []
    append = cur_rows.append
    for file in file_names:
        pf = pq.read_table(file)
        log.info('process file: {}'.format(file), save_to_file=True)

        for prompt, response in progress.track(zip(pf['INSTRUCTION'], pf['RESPONSE']), total=pf.num_rows):
            all_cnt += 1
            prompt, response = prompt.as_py(), response.as_py()
            
            prompt = process_function(prompt)
            response = process_function(response)

            if len(prompt) < prompt_less_word or len(response) < response_less_word:
                continue
            
            keep_cnt += 1
            write_dict = {
                'prompt': prompt,
                'response': response,
            }
            append(write_dict)

            if len(cur_rows) >= group_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(save_file, df)
                cur_rows = []
                append = cur_rows.append
            
    # end for 
    if len(cur_rows) > 0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(save_file, df)
        cur_rows = []

    log.info('save file to: {}, 全部数据共{}行，清洗后剩余{}行'.format(save_file, all_cnt, keep_cnt), save_to_file=True)


def process_belle_knowledge_enhanced_dataset(response_less_words: int=15, group_cnt: int=10000) -> None:
    '''
    处理belle开源的知识增强数据集
    '''
    file_names = [
        '/data/raw_data/bell_open_source/train_2M_CN.json',
        '/data/raw_data/bell_open_source/train_0.8M_CN.json',
        '/data/raw_data/bell_open_source/Belle_open_source_1M.json',
    ]

    save_file = PROJECT_ROOT + '/data/my_data/my_belll_3M_cn.parquet'

    # 后续append写入，存在文件先删除
    if exists(save_file): 
        assert delete_file(save_file)

    def process_function(line: str) -> dict:
        '''
        每行的处理函数
        '''
        item = ujson.loads(line)
        prompt = item['instruction']
        response = item['output']

        # 剔除翻译任务
        if '翻译' in prompt or 'translate' in prompt.lower():
            return None
        
        # 删除表格类任务
        if '表格' in prompt or '-----' in prompt or '-----' in response:
            return None

        if len(response) < response_less_words:
            return None
        
        prompt = remove_duplicate_punctuation(prompt)
        response = remove_duplicate_punctuation(response)

        if len(response) < response_less_words:
            return None

        write_dict = {
            'prompt': prompt,
            'response': response
        }

        return write_dict

    for file in file_names:
        file = PROJECT_ROOT + file

        read_and_write_template(file, save_file, process_function)

def convert_wiki_to_simple_zh(buffer_size: int=10000) -> None:
    '''
    将繁体wiki转换为简体Wiki
    '''
    raw_zh_wiki_file = PROJECT_ROOT + '/data/raw_data/wiki.txt'
    save_zh_wiki_simple_file = PROJECT_ROOT + '/data/raw_data/wiki.simple.txt' 

    if exists(save_zh_wiki_simple_file): 
        assert delete_file(save_zh_wiki_simple_file)

    cc = OpenCC('t2s')
    cur_rows = []
    append = cur_rows.append
    def procees_line(line: str) -> str:
        '''
        处理一行文本
        '''
        # 将繁体转换为简体
        line = cc.convert(line)

        line = re.sub(r"\「|\」|\｢|\｣|\『|\』", '\"', line)  # 将「」｢｣『』这些符号替换成引号
        line = re.sub(r"\，\）|\；\）", '）', line)  # 罗德·法尼(Rod Dodji Fanni，）
        line = re.sub(r"\（\，|\(\，", '（', line)  # 阿魯拉·基馬(Alula Girma (，
        
        line = convert_en_punctuation_to_zh_punct(line) # 英文标点转换为中文标点
        line = remove_duplicate_punctuation(line)  # 删除中文空括号和重复的标点

        return line
    with progress.open(raw_zh_wiki_file, 'r', encoding='utf-8') as read_f:
        with open(save_zh_wiki_simple_file, 'a', encoding='utf-8') as write_f:
            for line in read_f:
                line = procees_line(line)
                if len(line.strip()) == 0: continue

                line = '{}\n'.format(line)
                append(line)

                if len(cur_rows) >= buffer_size:
                    write_f.writelines(cur_rows)
                    cur_rows = []
                    append = cur_rows.append
            
            if len(cur_rows) > 0:
                write_f.writelines(cur_rows)
                cur_rows = []
        

def process_zh_wiki_data_to_datset(groups_cnt: int=10000, max_len: int=512, seed: int=23333) -> None:
    '''
    将Wiki中文数转换为问答数据集
    wiki 下载地址：https://dumps.wikimedia.org/zhwiki/
    将下载的bz2文件转换为wiki.txt参考：https://github.com/apertium/WikiExtractor
    '''
    raw_zh_wiki_file = PROJECT_ROOT + '/data/raw_data/wiki.txt'
    zhwiki_simple_file = PROJECT_ROOT + '/data/my_data/wiki_zh_simple.parquet'

    # 删除已经存在的数据
    if exists(zhwiki_simple_file): 
        assert delete_file(zhwiki_simple_file)

    # 将繁体转换为简体
    cc = OpenCC('t2s')
    all_cnt, keep_cnt = 0, 0
    
    # 构造问题的前缀
    prompt_prefix = [
        '什么是{}？',
        '介绍一下{}',
        '介绍一下什么是{}',
        '写一篇关于{}的介绍',
        '{}是什么？',
        '你知道{}吗？',
        '生成关于{}的介绍',
        '我想知道关于{}的详细信息',
        '你了解{}吗？',
        '请解释一下{}',
        '对于{}，你有什么了解或看法吗？',
        '请告诉我关于{}的信息',
        '请简要描述一下{}',
        '请提供有关{}的一些详细信息',
        '能否解释一下{}是什么?',
        '请分享一些关于{}的背景知识',
        '请简要概括一下{}',
        '能给我一些关于{}的背景资料吗?',
        '有关{}的信息可以分享一下吗？',
        '你能告诉我{}是什么吗？',
    ]

    def procees_line(line: str) -> str:
        '''
        处理一行文本
        '''
        # 将繁体转换为简体
        line = cc.convert(line)

        line = re.sub(r"\「|\」|\｢|\｣|\『|\』", '\"', line)  # 将「」｢｣『』这些符号替换成引号
        line = re.sub(r"\，\）|\；\）", '）', line)  # 罗德·法尼(Rod Dodji Fanni，）
        line = re.sub(r"\（\，|\(\，", '（', line)  # 阿魯拉·基馬(Alula Girma (，
        
        line = convert_en_punctuation_to_zh_punct(line) # 英文标点转换为中文标点
        line = remove_duplicate_punctuation(line)  # 删除中文空括号和重复的标点

        return line
        
    np.random.seed(seed)
    choice = np.random.choice

    with progress.open(raw_zh_wiki_file, 'r', encoding='utf-8') as read_file:
        prompt = '' 
        response = '' 
        pre_line_len = 0
        cur_rows = []
        append = cur_rows.append
        for line in read_file:
            all_cnt += 1

            # prompt已经保存，但是仍有多余的行，这些行使得response的长度＞max_len，故跳过，不处理
            if len(prompt) == 0 and pre_line_len > 0:
                pre_line_len = len(line.strip())
                continue
            
            # 清洗一行
            line = procees_line(line)
            

            # 确定问题，pre_line_len是0，既是上一行是空行，则当前行是新的百科词条，设置为prompt
            if prompt == '' and line.endswith('：') and pre_line_len == 0:
                prompt = choice(prompt_prefix).format(line[0: -1])
                continue

            pre_line_len = len(line.strip())

            # 问题下来若干行为答案
            if prompt != '' and not line.endswith('：'):
                # 其实，pre_line_len已经是len(line.strip())了，如果len(line.strip())=0，既是当前行是0，则不管答案长度够不够，都需要保存了
                if len(response) + len(line) <= max_len and pre_line_len != 0: 
                    response = '{}{}'.format(response, line)
                elif len(response) + len(line) > max_len or pre_line_len == 0:
                    # 长度超了或者当前的百科已经结束，保存一条样例
                    keep_cnt += 1
                    response = '{}{}'.format(response, line)
                    append({'prompt': prompt, 'response': ''.join(response[0: max_len])})
                    prompt = ''
                    response = ''

            # =groups_cnt保存到文件
            if len(cur_rows) >= groups_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(zhwiki_simple_file, df)
                cur_rows = []
                append = cur_rows.append

        # end for
        if len(prompt) > 0 and len(response) > 0:
            keep_cnt += 1
            append({'prompt': prompt, 'response': response})

        if len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(zhwiki_simple_file, df)
            cur_rows = []

    log.info("merge into file: {}, 全部数据共{}行，清洗后剩余{}行".format(zhwiki_simple_file, all_cnt, keep_cnt), save_to_file=True)



def merge_dataset_as_single_file(groups_cnt: int=50000, max_len: int=512, min_len: int=3, cut_max_len: bool=False) -> None:
    '''
    将多个数据集合并为一个数据集
    '''
    from_parquet_files = get_path_of_suffix_files(PROJECT_ROOT + '/data/my_data', '.parquet')

    save_file = PROJECT_ROOT + '/data/my_dataset.parquet'

    # 后续append写入，存在文件先删除
    if exists(save_file): 
        assert delete_file(save_file)

    cur_rows = []
    append = cur_rows.append
    
    all_cnt, keep_cnt = 0, 0
    for file in from_parquet_files:
        print('process file: {}'.format(file))

        parquet_table = pq.read_table(file)
     
        for prompt, response in progress.track(zip(parquet_table['prompt'], parquet_table['response']), total=parquet_table.num_rows):

            prompt, response = prompt.as_py(), response.as_py()
            all_cnt += 1

            if len(prompt) < min_len or len(response) < min_len:
                continue

            if cut_max_len and (len(prompt) > max_len or len(response) > max_len):
                prompt = prompt[0: max_len]
                response = response[0: max_len]

            keep_cnt += 1
            append({'prompt': prompt , 'response': response})

            if len(cur_rows) >= groups_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(save_file, df)
                cur_rows = []
                append = cur_rows.append
        
    # 处理末尾部分
    if len(cur_rows) > 0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(save_file, df)
        cur_rows = []

    log.info("merge into file: {}, 全部数据共{}行，清洗后剩余{}行".format(save_file, all_cnt, keep_cnt), save_to_file=True)


def remove_dataset_duplicate_rows(groups_cnt: int=50000) -> None:
    '''
    使用mini_hash删除数据集中重复的部分
    '''
    from_parquet_files = PROJECT_ROOT + '/data/my_dataset.parquet'

    save_file = PROJECT_ROOT + '/data/my_dataset_no_dulpticates.parquet'

    # 后续append写入，存在文件先删除
    if exists(save_file): 
        assert delete_file(save_file)

    cur_rows = []
    all_cnt, keep_cnt = 0, 0
    row_index = -1
    drop_dataset_duplicate = DropDatasetDuplicate(threshold=0.85, num_perm=256)
    
    parquet_table = pq.read_table(from_parquet_files)
    all_cnt = parquet_table.num_rows

    # 先顺序遍历获取哪些行是重复的
    for prompt, response in progress.track(zip(parquet_table['prompt'], parquet_table['response']), total=parquet_table.num_rows):
        row_index += 1

        doc = f"{prompt.as_py()}{response.as_py()}"
        drop_dataset_duplicate.add_doc(index=row_index, doc=doc)

    row_index = -1
    need_to_drop_indexs = drop_dataset_duplicate.get_duplicate_indexs()

    # 再顺序遍历一遍，重复的行不添加到新的数据集
    for prompt, response in progress.track(zip(parquet_table['prompt'], parquet_table['response']), total=parquet_table.num_rows):
        row_index += 1  # 不管有没有跳过行, row_index都必须+1

        # 重复的行跳过
        if row_index in need_to_drop_indexs:
            continue

        cur_rows.append({'prompt': prompt.as_py() , 'response': response.as_py()})
        keep_cnt += 1

        if len(cur_rows) >= groups_cnt:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(save_file, df)
            cur_rows = []

    # 处理末尾部分
    if len(cur_rows) > 0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(save_file, df)

    log.info("merge into file: {}, 全部数据共{}行，文档去重后剩余{}行".format(save_file, all_cnt, keep_cnt), save_to_file=True)

def shuffle_parquet_dataset(parquet_file: str, shuffle_file: str, seed: int=23333, groups_cnt: int=65536) -> None:
    '''
    打乱一个parquet文件数据集
    '''
    if not exists(parquet_file):
        raise Exception('can not find parquet file: {}'.format(parquet_file))
    
    print('start shuffle...')
    pf =  pq.read_table(parquet_file)
    df = pf.to_pandas()
    df = df.sample(frac=1.0, replace=False, random_state=seed, axis=0)
    
    if exists(shuffle_file): 
        assert delete_file(shuffle_file)
    
    # 分块写入parquet，否则小内存读取直接OOM
    n = len(df)
    for i in range(0, n, groups_cnt):
        cur_group_df = df[i: i + groups_cnt]
        write_single_parquet_file(shuffle_file, cur_group_df)

def count_my_json_data() -> None:
    '''
    统计目前的所有数据集数据量
    '''
    my_data_files = get_path_of_suffix_files(PROJECT_ROOT + '/data/my_data', '.json')
    result = [['file_name', 'count']]
    all_cnt = 0
    for file in my_data_files:
        file_name = file.split('/')[-1]
        cur_cnt = 0
        with progress.open(file, 'r', encoding='utf-8') as f:
            for _ in f:
                cur_cnt += 1
        
        all_cnt += cur_cnt
        result.append([file_name, cur_cnt])
    
    result.append(['汇总', all_cnt])

    log.info(str(result), save_to_file=True)

    console = Console()
    table = Table(show_header=True, show_lines=True,)

    for col in result[0]:
        table.add_column(col)
    for i in range(1, len(result)): # 跳过表头
        table.add_row(str(result[i][0]), str(result[i][1]))

    console.print(table)


def count_my_parquet_data(parquet_file: str=None) -> None:
    '''
    统计dir目录下所有parquet数据集数据量
    '''
    my_data_files = []

    if not parquet_file:
        my_data_files = get_path_of_suffix_files(PROJECT_ROOT + '/data/my_data', '.parquet')
    elif isdir(parquet_file):
        my_data_files = get_path_of_suffix_files(parquet_file, '.parquet')
    elif parquet_file.endswith('.parquet'):
        my_data_files = [parquet_file]
        

    result = [['file_name', 'count']]
    all_cnt = 0
    for file in my_data_files:
        file_name = file.split('/')[-1]
        cur_cnt = 0
        pf = ParquetFile(file)

        for pf_chunk in pf:
            cur_cnt += pf_chunk.info['rows']
        
        all_cnt += cur_cnt
        result.append([file_name, cur_cnt])
    
    result.append(['汇总', all_cnt])

    log.info(str(result), save_to_file=True)

    console = Console()
    table = Table(show_header=True, show_lines=True,)

    for col in result[0]:
        table.add_column(col)
    for i in range(1, len(result)): # 跳过表头
        table.add_row(str(result[i][0]), str(result[i][1]))

    console.print(table)    


def split_train_valid_test_datasets(source_parquet_file: str, max_len: int=320, seed: int=23333, train_ratio: float=0.91, test_ratio: float=0.0875, valid_ratio: float=0.0025, groups_cnt: int=50000) -> None:
    '''
    将原始数据拆分为训练集、测试集和验证集
    '''
    assert train_ratio + test_ratio + valid_ratio == 1.0

    train_parquet_file = PROJECT_ROOT + '/data/my_train_dataset.parquet'
    test_parquet_file = PROJECT_ROOT + '/data/my_test_dataset.parquet'
    valid_parquet_file = PROJECT_ROOT + '/data/my_valid_dataset.parquet'

    if exists(train_parquet_file): assert delete_file(train_parquet_file)
    if exists(test_parquet_file): assert delete_file(test_parquet_file)
    if exists(valid_parquet_file): assert delete_file(valid_parquet_file)

    np.random.seed(seed)

    train, test, valid = [], [], []

    parquet_table =  pq.read_table(source_parquet_file)

    for prompt, response in progress.track(zip(parquet_table['prompt'], parquet_table['response']), total=parquet_table.num_rows):
        
        prompt, response = prompt.as_py(), response.as_py()
        rand = np.random.random()
        cur_data = {'prompt': ''.join(prompt[0: max_len]) , 'response': ''.join(response[0: max_len])}

        if 0 <= rand < train_ratio:
            train.append(cur_data)
        elif train_ratio <= rand  < train_ratio + test_ratio:
            test.append(cur_data)
        else:
            valid.append(cur_data)
        
        if len(train) >= groups_cnt:
            write_single_parquet_file(train_parquet_file, pd.DataFrame(train))
            train = []
        
        if len(test) >= groups_cnt:
            write_single_parquet_file(test_parquet_file, pd.DataFrame(test))
            test = []
        
        if len(valid) >= groups_cnt:
            write_single_parquet_file(valid_parquet_file, pd.DataFrame(valid))
            valid = []
                

    if len(train) > 0:
        write_single_parquet_file(train_parquet_file, pd.DataFrame(train))
        train = []
    
    if len(test) > 0:
        write_single_parquet_file(test_parquet_file, pd.DataFrame(test))
        test = []
    
    if len(valid) > 0:
        write_single_parquet_file(valid_parquet_file, pd.DataFrame(valid))
        valid = []

def parquet_to_text(sep='[SEP]', buffer_size: int=50000) -> None:
    '''
    将parquet文件转换为txt预料，句子之间用sep隔开
    txt文件用于训练tokenizer，使用huggingface的BPE训练会导致OOM
    '''
    parquet_file = PROJECT_ROOT + '/data/my_dataset.parquet'
    txt_file = PROJECT_ROOT + '/data/my_corpus.txt'

    if exists(txt_file): 
        assert delete_file(txt_file)

    source_pf = ParquetFile(parquet_file)
    cur_rows = []
    append = cur_rows.append
    with open(txt_file, 'a', encoding='utf-8') as f_write:
        for pf_chunk in progress.track(source_pf):
            for rows in pf_chunk.iter_row_groups():
                for prompt, response in zip(rows['prompt'], rows['response']):
                    append(prompt + sep + response + sep + '\n')

                    if len(cur_rows) >= buffer_size:
                        f_write.writelines(cur_rows)
                        cur_rows = []
                        append = cur_rows.append
                       
        # end for
        if len(cur_rows) > 0:
            f_write.writelines(cur_rows)
            cur_rows = []

def parquet_to_json() -> None:
    '''
    将parquet文件转换为json
    '''
    parquet_file = PROJECT_ROOT + '/data/my_finetune_data_zh.parquet'
    json_file = PROJECT_ROOT + '/data/sft_train.json'

    if exists(json_file): 
        assert delete_file(json_file)

    source_pf = ParquetFile(parquet_file)
    cur_rows = []
    append = cur_rows.append
   
    for pf_chunk in progress.track(source_pf):
        for rows in pf_chunk.iter_row_groups():
            for prompt, response in zip(rows['prompt'], rows['response']):
                if len(response) == 0 or len(prompt) == 0: continue
                append({
                    'prompt': str(prompt),
                    'response': str(response),
                })

    with open(json_file, 'w', encoding='utf-8') as f:
        ujson.dump(cur_rows, f, indent=4, ensure_ascii=False)

def dataset_length_cnt() -> None:

    dataset_file = PROJECT_ROOT +  '/data/my_dataset.shuffle.parquet'
    parquet_table = pq.read_table(dataset_file)

    que_len_dict, ans_len_dict = defaultdict(int), defaultdict(int)
    
    for prompt, response in progress.track(zip(parquet_table['prompt'], parquet_table['response']), total=parquet_table.num_rows):

        prompt, response = prompt.as_py(), response.as_py()

        que_len_dict[len(prompt)] += 1
        ans_len_dict[len(response)] += 1

    que_len, ans_len = [], []
    for k, v in que_len_dict.items():
        que_len.append([k, v])
    for k, v in ans_len_dict.items():
        ans_len.append([k, v])

    def gather_gt_x(array: list[tuple], x: int=512) -> list:
        '''
        长度大于x的合并在一起
        '''
        new_array = []
        gt_x_cnt = 0
        for item in array:
            if item[0] < x:
                new_array.append([item[0], item[1]])
            else:
                gt_x_cnt += item[1]
        new_array.append([x, gt_x_cnt])

        return new_array
    
    max_len = 512
    ans_list = gather_gt_x(ans_len, max_len)
    ans_list.sort(key=lambda x: x[0])
    que_list = gather_gt_x(que_len, max_len)
    que_list.sort(key=lambda x: x[0])
    
    ans_pd = pd.DataFrame(ans_list, columns=['length', 'count'])
    que_pd = pd.DataFrame(que_list, columns=['length', 'count'])

    def plot_sub_bar(plt, x, y, title: str, color: str='g') ->None:
        plt.bar(x, y, color=color, label='sample count')
        plt.ticklabel_format(style='sci',scilimits=(0,0), axis='y')
        plt.legend()
        plt.xlabel('length')
        plt.ylabel('count')
        plt.title(title)

    plt.figure(figsize=(10, 10),dpi=200)
    plt.subplot(2, 2, 1)
    plot_sub_bar(plt, que_pd['length'], que_pd['count'], title='prompt length', color='c')

    plt.subplot(2, 2, 2)
    plot_sub_bar(plt, ans_pd['length'], ans_pd['count'], title='response length', color='g')

    le512_pd = ans_pd[ans_pd['length'] < 512]
    plt.subplot(2, 2, 3)
    plot_sub_bar(plt, le512_pd['length'], le512_pd['count'], title='response length < 512', color='limegreen')

    le320_pd = ans_pd[ans_pd['length'] < 320]
    plt.subplot(2, 2, 4)
    plot_sub_bar(plt, le320_pd['length'], le320_pd['count'], title='response length < 320', color='limegreen')

    plt.savefig(PROJECT_ROOT +  '/img/sentence_length.png')
    plt.show()

def process_belle_knowledge_enhanced_dataset_for_finetune(max_len: int=320, group_cnt: int=50000) -> None:
    '''
    处理belle开源的知识增强数据集
    '''
    file_names = [
        '/data/raw_data/bell_open_source/Belle_open_source_0.5M.json',
        '/data/raw_data/bell_open_source/train_conv_2.json',
        '/data/raw_data/bell_open_source/generated_chat_0.4M.json',
    ]

    save_file = PROJECT_ROOT + '/data/my_finetune_data_zh.parquet'

    # 后续append写入，存在文件先删除
    if exists(save_file): 
        assert delete_file(save_file)

    def process_function(line: str) -> dict:
        '''
        每行的处理函数
        '''
        item = ujson.loads(line)
        prompt = item['instruction']
        response = item['output']

        # 剔除翻译任务
        if 'translate' in prompt.lower(): return None
        for word in ('翻译', '英译', '译英', '中译',  '译中', '汉译', '译汉'):
            if word in prompt:
                return None
        
        # 删除表格类任务
        if '表格' in prompt or '-----' in prompt or '-----' in response:
            return None

        if len(prompt) > max_len or len(response) > max_len:
            return None

        write_dict = {
            'prompt': prompt,
            'response': response
        }

        return write_dict

    for file in file_names:
        file = PROJECT_ROOT + file

        read_and_write_template(file, save_file, process_function)


if __name__ == '__main__':

    processed_file_dir = PROJECT_ROOT + '/data/my_data'
    if not exists(processed_file_dir):
        mkdir(processed_file_dir)
    
    # 注释了，不重复处理
    # 1.
    # process_web_text(keep_start=5, response_less_word=15)

    # 2.
    # process_bake_qa(response_less_word=15)

    # 3.
    # process_chinese_medical_datasets(response_less_word=15)

    # 4. 金融问答数据集质量太差了
    # process_finace_dataset(prompt_less_word=10, response_less_word=15)

    # 5.
    # process_zhihu_kol_dataset(prompt_less_word=4, response_less_word=10)

    # 6.
    # process_belle_knowledge_enhanced_dataset(response_less_words=5)

    # convert_wiki_to_simple_zh()

    # 7.
    # process_zh_wiki_data_to_datset(groups_cnt=10000, max_len=512)

    #=================================================================

    # merge
    # merge_dataset_as_single_file(groups_cnt=50000, min_len=3, max_len=512, cut_max_len=True)
        
    
    remove_dataset_duplicate_rows(groups_cnt=50000)

    # # shuffle
    # shuffle_parquet_dataset(
    #     parquet_file=PROJECT_ROOT + '/data/my_dataset.parquet', 
    #     shuffle_file=PROJECT_ROOT + '/data/my_dataset.shuffle.parquet',  
    #     seed=23333
    # )

    # split train validated and test
    # split_train_valid_test_datasets(
    #         source_parquet_file=PROJECT_ROOT + '/data/my_dataset.shuffle.parquet',
    #         max_len=320, 
    #         groups_cnt=50000
    #     )

    # parquet_to_text()

    # count_my_parquet_data(PROJECT_ROOT + '/data/my_dataset.parquet')

    # dataset_length_cnt()

    # process_belle_knowledge_enhanced_dataset_for_finetune(max_len=320, group_cnt=50000)

    # count_my_parquet_data(PROJECT_ROOT + '/data/')

    parquet_to_json()
    # count_my_json_data()


