import ujson
import re
from os.path import dirname, abspath, exists
from os import remove, mkdir, walk
import time
import codecs, csv
from rich import progress

from logger import Logger



log = Logger('data_process', save2file=True, file_name='raw_data_process.log').get_logger()

ROOT_PATH = abspath(dirname(dirname(__file__))) + '/'

punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：\n")

def remove_duplicate_punctuation(sentence: str) -> str:
    '''
    删除句子中重复的标点符号
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


def read_and_write_template(read_file: str, write_to_file: str, call_back: object) -> None:
    '''
    处理数据读写模板，需要提供一个回调函数call_back，
    read_file: 原始数据文件
    write_to_file：处理后的要保存数据文件
    call_back：函数输入一个字符串，输出一个处理后的字典dict，如果输入的字符串为无效数据，请返回None
    如：
    >>> def call_back(inputs: str) -> dict:
    >>>     if check(inputs) not valid:
    >>>         return None
    >>>    
    >>>     do something for inputs
    >>>
    >>>     my_dict = {
    >>>             'question': inputs['q'],
    >>>             'answer': inputs['a1'] + inputs['a2'],
    >>>             ...
    >>>         }
    >>>     return my_dict
    '''

    # 因为后面要append写入，如果已经存在处理后的文件，先删除
    if exists(write_to_file):
        remove(write_to_file)
    
    log.info('process file:{}'.format(read_file))
    start = time.time()
    
    raw_line_cnt = 0
    keep_line_cnt = 0
    
    with open(write_to_file, 'a', encoding='utf-8') as f_write:
        with progress.open(read_file, 'r', encoding='utf-8') as f_read:
            for line in f_read:
                try:
                    raw_line_cnt += 1

                    write_obj = call_back(line)

                    if write_obj is None: continue

                    keep_line_cnt += 1
                    
                    # ujson.dump(write_obj, f_write, indent=4, ensure_ascii=False)
                    ujson.dump(write_obj, f_write,  ensure_ascii=False,)
                    f_write.write('\n')

                except Exception as e:
                    log.error('处理文件异常：{}, content:{}'.format(str(e), line))
                    raise e
        
    end = time.time()

    log.info('原始文件:{}，共{}行，处理后剩余{}行，保存到文件：{}。耗时：{:.6}s'\
                .format(read_file, raw_line_cnt, keep_line_cnt, write_to_file, end - start))

def process_web_text(keep_start: int=5, answer_less_word: int=10) -> None:
    '''
    处理425万社区问答webtext2019zh知识类数据集
    keep_start: 只保留点赞数大于keep_start的问答
    answer_less_word: 答案至少要有answer_less_word个字
    '''
    file_names = [
        'data/raw_data/web_text_zh_test.json',
        'data/raw_data/web_text_zh_train.json',
        'data/raw_data/web_text_zh_valid.json',
    ]

    save_file_names = [
        'data/my_data/my_web_text_zh_test.json',
        'data/my_data/my_web_text_zh_train.json',
        'data/my_data/my_web_text_zh_valid.json',
    ]

    def process_function(line: str) -> dict:
        item = ujson.loads(line)

        if item['star'] < keep_start or len(item['content']) <= answer_less_word: 
            return None

        # 数据清洗
        # 去除重复的标点符号
        question = remove_duplicate_punctuation(item['title'])
        answer = remove_duplicate_punctuation(item['content'])
        write_obj = {
            "question": question,
            "answer": answer,
            "star": item['star']
        }
        return write_obj

    for i, file_name in enumerate(file_names):
        read_file = ROOT_PATH + file_name
        write_file = ROOT_PATH + save_file_names[i]
        
        read_and_write_template(read_file, write_file, process_function)
                
                 
        
def process_bake_qa(answer_less_word: int=15) -> None:
    '''
    处理147万百度知道知识类数据集

    '''
    file_names = [
        'data/raw_data/baike_qa_train.json',
        'data/raw_data/baike_qa_valid.json',
    ]

    save_file_names = [
        'data/my_data/my_baike_qa_train.json',
        'data/my_data/my_baike_qa_valid.json',
    ]

    def process_function(line: str) -> dict:
        item = ujson.loads(line)

        if len(item['answer']) <= answer_less_word: 
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
        if len(question) <= 3 or len(answer) <= answer_less_word:
            return None
        
        write_obj = {
                "question": question,
                "answer": answer,
            }

        return write_obj

    for i, file_name in enumerate(file_names):
        read_file = ROOT_PATH + file_name
        write_file = ROOT_PATH + save_file_names[i]

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
    raw_dataset_dir = ROOT_PATH + 'data/raw_data/chinese_medical_dialogue_datasets'
    
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
        file_name = file_name.split('/')[-1][0: -(len(suffix))] + '.json'
        file_name = ROOT_PATH  + 'data/my_data/' + file_name
        save_files.append(file_name)
    
    def process_function(line: str) -> dict:
        # department,title,ask,answer
        item = line.split(',') # csv文件逗号分割
        if len(item) < 4:
            print(item)
            return None

        if len(item[3]) <= answer_less_word: 
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
        if len(question) <= 3 or len(answer) <= answer_less_word:
            return None
        
        write_obj = {
                "question": question,
                "answer": answer,
            }

        return write_obj

    for i, file_name in enumerate(raw_data_files):
        read_file = file_name
        write_file = save_files[i]

        read_and_write_template(read_file, write_file, process_function)


def process_finace_dataset(question_less_word: int=10, answer_less_word: int=15) -> None:
    '''
    处理金融问答数据集
    '''
    finace_data_file = ROOT_PATH + 'data/raw_data/financezhidao_filter.csv'
    
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
        if len(question) <= question_less_word or len(answer) <= answer_less_word:
            return None
        
        write_obj = {
                "question": question,
                "answer": answer,
            }

        return write_obj

  
    read_file = finace_data_file[0: -4] + suffix
    write_file = ROOT_PATH + 'data/my_data/' + read_file.split('/')[-1][0: -(len(suffix))] + '.json'

    read_and_write_template(read_file, write_file, process_function)

if __name__ == '__main__':

    processed_file_dir = ROOT_PATH + '/data/my_data'
    if not exists(processed_file_dir):
        mkdir(processed_file_dir)
    
    # 注释了，不重复处理
    # 1.
    # process_web_text(keep_start=10, answer_less_word=15)

    # 2.
    # process_bake_qa(answer_less_word=15)

    # 3.
    # process_chinese_medical_datasets(answer_less_word=15)

    # 4.
    process_finace_dataset(question_less_word=10, answer_less_word=15)


