import ujson
import re
from os.path import dirname, abspath, exists, isdir
from os import remove, mkdir, walk
import time
import codecs, csv
import pandas as pd 
import numpy as np
from rich import progress
from rich.table import Table
from rich.console import Console
from fastparquet import ParquetFile, write
from opencc import OpenCC
from tokenizers import Tokenizer

# import sys
# sys.path.extend(['.','..'])

from logger import Logger
from config import PROJECT_ROOT

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
    >>>             'question': inputs['q'],
    >>>             'answer': inputs['a1'] + inputs['a2'],
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

def process_web_text(keep_start: int=5, answer_less_word: int=10) -> None:
    '''
    处理425万社区问答webtext2019zh知识类数据集
    keep_start: 只保留点赞数大于keep_start的问答
    answer_less_word: 答案至少要有answer_less_word个字
    '''
    file_names = [
        '/data/raw_data/web_text_zh_test.json',
        '/data/raw_data/web_text_zh_train.json',
        '/data/raw_data/web_text_zh_valid.json',
    ]

    save_file_names = [
        '/data/my_data/my_web_text_zh_test.parquet',
        '/data/my_data/my_web_text_zh_train.parquet',
        '/data/my_data/my_web_text_zh_valid.parquet',
    ]

    def process_function(line: str) -> dict:
        item = ujson.loads(line)

        if item['star'] < keep_start or len(item['content']) < answer_less_word: 
            return None

        # 数据清洗
        # 去除重复的标点符号
        question = remove_duplicate_punctuation(item['title'])
        answer = remove_duplicate_punctuation(item['content'])
        write_dict = {
            "question": question,
            "answer": answer,
            "star": item['star']
        }
        return write_dict

    for i, file_name in enumerate(file_names):
        read_file = PROJECT_ROOT + file_name
        write_file = PROJECT_ROOT + save_file_names[i]
        
        # 后续append写入，存在文件先删除
        if exists(write_file): 
            assert delete_file(write_file)

        read_and_write_template(read_file, write_file, process_function)
                
                 
        
def process_bake_qa(answer_less_word: int=15) -> None:
    '''
    处理147万百度知道知识类数据集

    '''
    file_names = [
        '/data/raw_data/baike_qa_train.json',
        '/data/raw_data/baike_qa_valid.json',
    ]

    save_file_names = [
        '/data/my_data/my_baike_qa_train.parquet',
        '/data/my_data/my_baike_qa_valid.parquet',
    ]

    def process_function(line: str) -> dict:
        item = ujson.loads(line)

        if len(item['answer']) < answer_less_word: 
            return None

        # 数据清洗
        question = ''
        if get_sentences_dice_similarity(item['title'], item['desc']) >= 0.90:
            # title 和desc 相似度过高，只用title作为问题
            question = item['title']
        else:
            # title 和desc拼接形成问题
            question = "{}{}".format(item['title'], item['desc'])

        # 删除\r
        question = question.replace('\r','') 

        # 删除重复的标点符号
        question = remove_duplicate_punctuation(question)

        # 去除重复的标点符号
        answer = item['answer'].replace('\r','')
        answer = remove_duplicate_punctuation(answer)

        # 剔除问题和答案过短的数据
        if len(question) < 3 or len(answer) < answer_less_word:
            return None
        
        write_dict = {
                "question": question,
                "answer": answer,
            }

        return write_dict

    for i, file_name in enumerate(file_names):
        read_file = PROJECT_ROOT + file_name
        write_file = PROJECT_ROOT + save_file_names[i]
        
        # 后续append写入，存在文件先删除
        if exists(write_file): 
            assert delete_file(write_file)

        read_and_write_template(read_file, write_file, process_function)


def get_path_of_suffix_files(root: str, suffix: str) -> list:
    '''
        获取指定目录下下指定后缀的所有文件的绝对路径
    '''
    suffix_files = []
    for root, _, files in walk(root):
        for file in files:
            if file.endswith(suffix):
                suffix_files.append('{}/{}'.format(root, file))
                            
    return suffix_files  
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

def process_chinese_medical_datasets(answer_less_word: int=15) -> None:
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
    save_files = []
    for file_name in raw_data_files:
        file_name = file_name.split('/')[-1][0: -(len(suffix))] + '.parquet'
        file_name = PROJECT_ROOT  + '/data/my_data/' + file_name
        save_files.append(file_name)
    
    def process_function(line: str) -> dict:
        # department,title,ask,answer
        item = line.split(',') # csv文件逗号分割
        if len(item) < 4:
            print(item)
            return None

        if len(item[3]) < answer_less_word: 
            return None

        # 数据清洗
        question = ''
        if get_sentences_dice_similarity(item[1], item[2]) >= 0.90:
            # title 和ask 相似度过高，只用ask作为问题
            question = item[2]
        else:
            # title 和 ask 拼接形成问题
            question = "{}{}".format(item[1], item[2])

        # 删除\r
        question = question.replace('\r','') 

        # 删除重复的标点符号
        question = remove_duplicate_punctuation(question)

        # 去除重复的标点符号
        answer = ''.join(item[3: ]).replace('\r','')
        answer = remove_duplicate_punctuation(answer)

        # 剔除问题和答案过短的数据
        if len(question) < 3 or len(answer) < answer_less_word:
            return None
        
        write_dict = {
                "question": question,
                "answer": answer,
            }

        return write_dict

    for i, file_name in enumerate(raw_data_files):
        read_file = file_name
        write_file = save_files[i]
        
        # 后续append写入，存在文件先删除
        if exists(write_file): 
            assert delete_file(write_file)

        read_and_write_template(read_file, write_file, process_function)


def process_finace_dataset(question_less_word: int=10, answer_less_word: int=15) -> None:
    '''
    处理金融问答数据集
    '''
    finace_data_file = PROJECT_ROOT + '/data/raw_data/financezhidao_filter.csv'
    
    suffix = '.repaired.csv'
    if not exists(finace_data_file[0: -4] + suffix):
        repair_line_error_csv_file(finace_data_file, save_suffix=suffix, read_encoding='utf-8')

    
    def process_function(line: str) -> dict:
        # title,question,reply,is_best
        item = line.split(',') # csv文件逗号分割
        if len(item) < 4:
            print(item)
            return None

        if len(item[0]) + len(item[1]) < question_less_word or len(item[2]) < answer_less_word: 
            return None

        # 数据清洗
        question = ''
        if get_sentences_dice_similarity(item[0], item[1]) >= 0.90:
            # title 和question 相似度过高，只用最长的作为问题
            question = item[0] if len(item[0]) > len(item[0]) else item[1]
        else:
            # title 和 ask 拼接形成问题
            question = "{}{}".format(item[0], item[1])

        # 删除\r
        question = question.replace('\r','') 

        # 删除重复的标点符号
        question = remove_duplicate_punctuation(question)

        # 去除重复的标点符号
        answer = ''.join(item[2]).replace('\r','')
        answer = remove_duplicate_punctuation(answer)

        # 剔除问题和答案过短的数据
        if len(question) < question_less_word or len(answer) < answer_less_word:
            return None
        
        write_obj = {
                "question": question,
                "answer": answer,
            }

        return write_obj

  
    read_file = finace_data_file[0: -4] + suffix
    write_file = PROJECT_ROOT + '/data/my_data/' + read_file.split('/')[-1][0: -(len(suffix))] + '.parquet'

    # 后续append写入，存在文件先删除
    if exists(write_file): 
        assert delete_file(write_file)

    read_and_write_template(read_file, write_file, process_function)


def process_zhihu_kol_dataset(question_less_word: int=4, answer_less_word: int=10, group_cnt: int=10000) -> None:
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
        pf = ParquetFile(file)
        log.info('process file: {}'.format(file), save_to_file=True)

        for pf_chunk in progress.track(pf): # pf分块
            for rows in pf_chunk.iter_row_groups():
                for question, answer in zip(rows['INSTRUCTION'], rows['RESPONSE']):
                    all_cnt += 1
                    
                    question = process_function(question)
                    answer = process_function(answer)

                    if len(question) < question_less_word or len(answer) < answer_less_word:
                        continue
                    
                    keep_cnt += 1
                    write_dict = {
                        'question': question,
                        'answer': answer,
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


def process_belle_knowledge_enhanced_data_set(answer_less_words: int=15, group_cnt: int=10000) -> None:
    '''
    处理belle开源的知识增强数据集
    '''
    file_names = [
        'data/raw_data/bell_open_source/train_2M_CN.json',
        'data/raw_data/bell_open_source/Belle_open_source_1M.json',
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
        question = item['instruction']
        answer = item['output']

        if len(answer) < answer_less_words:
            return None
        
        question = remove_duplicate_punctuation(question)
        answer = remove_duplicate_punctuation(answer)

        if len(answer) < answer_less_words:
            return None

        write_dict = {
            'question': question,
            'answer': answer
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
    question_prefix = [
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
        '对于{}，你有什么了解或看法的吗？',
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
        question = '' 
        answer = '' 
        pre_line_len = 0
        cur_rows = []
        append = cur_rows.append
        for line in read_file:
            all_cnt += 1

            # question已经保存，但是仍有多余的行，这些行使得answer的长度＞max_len，故跳过，不处理
            if len(question) == 0 and pre_line_len > 0:
                pre_line_len = len(line.strip())
                continue
            
            # 清洗一行
            line = procees_line(line)
            

            # 确定问题，pre_line_len是0，既是上一行是空行，则当前行是新的百科词条，设置为question
            if question == '' and line.endswith('：') and pre_line_len == 0:
                question = choice(question_prefix).format(line[0: -1])
                continue

            pre_line_len = len(line.strip())

            # 问题下来若干行为答案
            if question != '' and not line.endswith('：'):
                # 其实，pre_line_len已经是len(line.strip())了，如果len(line.strip())=0，既是当前行是0，则不管答案长度够不够，都需要保存了
                if len(answer) + len(line) <= max_len and pre_line_len != 0: 
                    answer += line
                elif len(answer) + len(line) > max_len or pre_line_len == 0:
                    # 长度超了或者当前的百科已经结束，保存一条样例
                    keep_cnt += 1
                    append({'question': question, 'answer': answer})
                    question = ''
                    answer = ''

            # =groups_cnt保存到文件
            if len(cur_rows) >= groups_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(zhwiki_simple_file, df)
                cur_rows = []
                append = cur_rows.append

        # end for
        if len(question) > 0 and len(answer) > 0:
            keep_cnt += 1
            append({'question': question, 'answer': answer})

        if len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(zhwiki_simple_file, df)
            cur_rows = []

    log.info("merge into file: {}, 全部数据共{}行，清洗后剩余{}行".format(zhwiki_simple_file, all_cnt, keep_cnt), save_to_file=True)



def merge_dataset_as_single_file(groups_cnt: int=50000, max_len: int=512, cut_max_len: bool=False) -> None:
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

        pf = ParquetFile(file)
        for pf_chunk in progress.track(pf):
            for rows in pf_chunk.iter_row_groups():
                for question, answer in zip(rows['question'], rows['answer']):
                    all_cnt += 1

                    if cut_max_len and (len(question) > max_len or len(answer) > max_len):
                        question = question[0: max_len]
                        answer = answer[0: max_len]

                    keep_cnt += 1
                    append({'question': question , 'answer': answer})

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

def shuffle_parquet_dataset(parquet_file: str, shuffle_file: str, seed: int=23333) -> None:
    '''
    打乱一个parquet文件数据集
    '''
    if not exists(parquet_file):
        raise Exception('can not find parquet file: {}'.format(parquet_file))
    
    print('start shuffle...')
    pf = ParquetFile(parquet_file)
    df = pf.to_pandas()
    df = df.sample(frac=1.0, replace=False, random_state=seed, axis=0)
    
    if exists(shuffle_file): 
        assert delete_file(shuffle_file)
    
    write_single_parquet_file(shuffle_file, df)

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

    source_pf = ParquetFile(source_parquet_file)
    for pf_chunk in progress.track(source_pf):
        for rows in pf_chunk.iter_row_groups():
            for question, answer in zip(rows['question'], rows['answer']):
                rand = np.random.random()

                cur_data = {'question': question[0: max_len] , 'answer': answer[0: max_len]}

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
                for question, answer in zip(rows['question'], rows['answer']):
                    append(question + sep + answer + sep + '\n')

                    if len(cur_rows) >= buffer_size:
                        f_write.writelines(cur_rows)
                        cur_rows = []
                        append = cur_rows.append
                       
        # end for
        if len(cur_rows) > 0:
            f_write.writelines(cur_rows)
            cur_rows = []

def text_dataset_to_ids_dataset(tokenizer_file: str, max_len: int=320, pad_token: str='[PAD]') -> None:
    '''
    将text字符全部转换为id， 从parquet文件加载list对象太慢了，这个函数没用
    '''

    text_files = [
        PROJECT_ROOT + '/data/my_train_dataset.parquet',
        PROJECT_ROOT + '/data/my_test_dataset.parquet',
        PROJECT_ROOT + '/data/my_valid_dataset.parquet',
    ]

    save_ids_files = [
        PROJECT_ROOT + '/data/my_train_ids_dataset.parquet',
        PROJECT_ROOT + '/data/my_test_ids_dataset.parquet',
        PROJECT_ROOT + '/data/my_valid_ids_dataset.parquet',
    ]

    # 询问删除旧的文件
    for ids_file in save_ids_files:
            if exists(ids_file): 
                assert delete_file(ids_file)

    tokenizer = Tokenizer.from_file(tokenizer_file)
    tokenizer.enable_padding(length=max_len)
    tokenizer.enable_truncation(max_length=max_len)
    encode_batch = tokenizer.encode_batch

    for i, txt_file in enumerate(text_files):
        
        log.info('processing file: {}'.format(txt_file))
        parquet_data = ParquetFile(txt_file)

        for pf_chunk in progress.track(parquet_data):
                for rows in pf_chunk.iter_row_groups(): #一个pf_chunk大概1万-5万条数据，视写入parquet时的配置定

                    # text to ids
                    question = encode_batch(rows['question'])
                    answer = encode_batch(rows['answer'])
                    
                    input_ids = [q.ids for q in question]
                    target_ids = [a.ids for a in answer]

                    df = pd.DataFrame({'input_ids': input_ids, 'target_ids': target_ids})
                    write_single_parquet_file(save_ids_files[i], df)


if __name__ == '__main__':

    processed_file_dir = PROJECT_ROOT + '/data/my_data'
    if not exists(processed_file_dir):
        mkdir(processed_file_dir)
    
    # 注释了，不重复处理
    # 1.
    # process_web_text(keep_start=5, answer_less_word=15)

    # 2.
    # process_bake_qa(answer_less_word=15)

    # 3.
    # process_chinese_medical_datasets(answer_less_word=15)

    # 4.
    # process_finace_dataset(question_less_word=10, answer_less_word=15)

    # 5.
    # process_zhihu_kol_dataset(question_less_word=4, answer_less_word=10)

    # 6.
    # process_belle_knowledge_enhanced_data_set(answer_less_words=15)

    # 7.
    # process_zh_wiki_data_to_datset(groups_cnt=10000, max_len=512)

    # convert_wiki_to_simple_zh()

    # merge
    # merge_dataset_as_single_file(groups_cnt=50000, cut_max_len=False)

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

    # 从parquet文件加载list太慢了
    # text_dataset_to_ids_dataset(tokenizer_file=PROJECT_ROOT + '/model_save/my_merged_tokenizer.json', max_len=320)

    # count_my_parquet_data(PROJECT_ROOT + '/data/my_dataset.parquet')
    count_my_parquet_data(PROJECT_ROOT + '/data/')


    # count_my_json_data()


