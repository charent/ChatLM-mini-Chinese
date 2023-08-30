import ujson
import re
from os import remove, mkdir
import time
from os.path import dirname, abspath, exists
from logger import Logger
from rich import progress


log = Logger('data_process', save2file=True, file_name='raw_data_process.log').get_logger()

DATA_PATH = abspath(dirname(dirname(__file__))) + '/'

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

    # 如果已经存在处理后的文件，先删除
    for file_name in save_file_names:
        save_path =  DATA_PATH + file_name
        if exists(save_path):
            remove(save_path)

    for i, file_name in enumerate(file_names):
        read_file = DATA_PATH + file_name
        write_file = DATA_PATH + save_file_names[i]

        log.info('process file:{}'.format(read_file))
        start = time.time()
        
        raw_line_cnt = 0
        keep_line_cnt = 0
       
        with open(write_file, 'a', encoding='utf-8') as f_write:

            with progress.open(read_file, 'r', encoding='utf-8') as f_read:
                for line in f_read:
                    try:
                        item = ujson.loads(line)
                        raw_line_cnt += 1

                        if item['star'] < keep_start or len(item['content']) <= answer_less_word: 
                            continue
                        
                        keep_line_cnt += 1

                        # 数据清洗
                        # 去除重复的标点符号
                        question = remove_duplicate_punctuation(item['title'])
                        answer = remove_duplicate_punctuation(item['content'])
                        write_obj = {
                            "question": question,
                            "answer": answer,
                            "star": item['star']
                        }
                        
                        # ujson.dump(write_obj, f_write, indent=4, ensure_ascii=False)
                        ujson.dump(write_obj, f_write,  ensure_ascii=False,)
                        f_write.write('\n')

                        # if keep_line_cnt >= 5:
                        #     break

                    except Exception as e:
                        log.error('处理文件异常：{}, content:{}'.format(str(e), line))
            
        end = time.time()

        log.info('原始文件:{}，共{}行，处理后剩余{}行，保存到文件：{}。耗时：{}s'\
                 .format(read_file, raw_line_cnt, keep_line_cnt, write_file, end - start))
        
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

    # 如果已经存在处理后的文件，先删除
    for file_name in save_file_names:
        save_path =  DATA_PATH + file_name
        if exists(save_path):
            remove(save_path)

    for i, file_name in enumerate(file_names):
        read_file = DATA_PATH + file_name
        write_file = DATA_PATH + save_file_names[i]

        log.info('process file:{}'.format(read_file))
        start = time.time()
        
        raw_line_cnt = 0
        keep_line_cnt = 0
       
        with open(write_file, 'a', encoding='utf-8') as f_write:

            with progress.open(read_file, 'r', encoding='utf-8') as f_read:
                for line in f_read:
                    try:
                        item = ujson.loads(line)
                        raw_line_cnt += 1

                        if len(item['answer']) <= answer_less_word: 
                            continue

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
                            continue

                        keep_line_cnt += 1

                        write_obj = {
                            "question": question,
                            "answer": answer,
                        }
                        
                        # ujson.dump(write_obj, f_write, indent=4, ensure_ascii=False)
                        ujson.dump(write_obj, f_write,  ensure_ascii=False,)
                        f_write.write('\n')

                        # if keep_line_cnt >= 5:
                        #     break

                    except Exception as e:
                        log.error('处理文件异常：{}, content:{}'.format(str(e.with_traceback()), line))
                        # raise e
            
        end = time.time()

        log.info('原始文件:{}，共{}行，处理后剩余{}行，保存到文件：{}。耗时：{}s'\
                 .format(read_file, raw_line_cnt, keep_line_cnt, write_file, end - start))
    
    
    
if __name__ == '__main__':

    processed_file_dir = DATA_PATH + '/data/my_data'
    if not exists(processed_file_dir):
        mkdir(processed_file_dir)
    
    # 注释了，不重复处理
    # 1.
    # process_web_text(keep_start=10, answer_less_word=15)

    # 2.
    # process_bake_qa(answer_less_word=15)



