import ujson
from os import remove
import time
from os.path import dirname, abspath, exists
from logger import Logger
from rich import progress


log = Logger('data_process', save2file=True, file_name='raw_data_process.log').get_logger()

RAW_DATA_PATH = abspath(dirname(dirname(__file__))) + '/'

punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：")

def remove_duplicate_punctuation(sentence: str) -> str:
    '''
    删除句子中重复的标点符号
    '''
    # 将空格替换为逗号
    sentence = sentence.replace(' ', '，')

    ans = ''
    n = len(sentence)
    p = 0
    while p < n:
        ans += sentence[p]

        while p + 1 < n and sentence[p] in punctuation and sentence[p + 1] in punctuation:
            p += 1
        p += 1

    return ans

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
        'data/raw_data/my_web_text_zh_test.json',
        'data/raw_data/my_web_text_zh_train.json',
        'data/raw_data/my_web_text_zh_valid.json',
    ]

    # 如果已经存在处理后的文件，先删除
    for file_name in save_file_names:
        save_path =  RAW_DATA_PATH + file_name
        if exists(save_path):
            remove(save_path)

    for i, file_name in enumerate(file_names):
        read_file = RAW_DATA_PATH + file_name
        write_file = RAW_DATA_PATH + save_file_names[i]

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
    
if __name__ == '__main__':
    process_web_text(keep_start=6)