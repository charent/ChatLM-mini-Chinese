import ujson
import codecs
import re
from rich import progress
import numpy as np 


def process_all_50_schemas(raw_schemas_file: str='./data/all_50_schemas', save_schemas_file: str=None) -> list[str]:
    '''
    获取prompt的关系列表
    '''
    lines = []
    with codecs.open(raw_schemas_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    scheme_list = []
    for line in lines:
        item = ujson.loads(line)
        scheme_list.append(
            item['predicate']
        ) 
    
    scheme_list = list(set(scheme_list))
    
    if save_schemas_file:
        with codecs.open(save_schemas_file, 'w', encoding='utf-8') as f:
            ujson.dump(f"{scheme_list}", f, indent=4, ensure_ascii=False)

    return scheme_list

def process_spo_list(text: str, spo_list: list, repair_song: bool=False):
    '''
    处理spo_list,处理成{subject: 'subject', subject_start: 0, subject_end:3, predicate: 'predicate', object: 'object', object_start: 5, object_end = 7}
    '''
    new_spo_list = []

    # 找出所有用书名号隔开的名字
    some_name = re.findall('《([^《》]*?)》', text)
    some_name = [n.strip() for n in some_name]
    
    # 歌曲和专辑
    song = []
    album = []
    for spo in spo_list:

        # 修正so的错误，删除前后的书名号
        s = spo['subject'].strip('《》').strip().lower()
        o = spo['object'].strip('《》').strip().lower()
        p = spo['predicate']
        
        # 如果s在找到的名字中，以正则找到的s为准，用in判等，
        # 如text: '《造梦者---dreamer》'，但是标注的s是'造梦者'
        for name in some_name:
            if s in name and text.count(s) == 1:
                s = name
        
        if repair_song:
            if p == '所属专辑':
                song.append(s)
                album.append(o)

        temp = dict()
        temp['s'] = s
        temp['p'] = spo['predicate']
        temp['o'] = o


        # 在text中找不到subject 或者 object，不要这条数据了
        if text.find(s) == -1 or text.find(o) == -1:
            continue

        new_spo_list.append(temp)
    
    if repair_song:
        ret_spo_list = []
        ps = ['歌手', '作词', '作曲']
        
        for spo in new_spo_list:
            s, p, o = spo['s'], spo['p'], spo['o']
            if p in ps and s in album and s not in song:
                continue
            ret_spo_list.append(spo)

        return ret_spo_list

    return new_spo_list


def process_data(raw_data_file: str, train_file_name: str, dev_file_name: str, keep_max_length: int=512, repair_song: bool=True, dev_size: int=1000) -> None:
    '''
    将原始的格式处理为prompt：resopnse的格式
    '''
    lines = []
    with codecs.open(raw_data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    my_raw_data = []

    schemas = process_all_50_schemas('./data/all_50_schemas')
    schemas = f"[{'，'.join(schemas)}]"
    for i, line in progress.track(enumerate(lines), total=len(lines)):
        
        tmp = ujson.decode(line)
        text = f"请抽取出给定句子中的所有三元组。给定句子：{tmp['text'].lower()}"
            
        spo_list = process_spo_list(tmp['text'].lower(), tmp['spo_list'], repair_song=repair_song)
        spo = f"{[(item['s'], item['p'], item['o']) for item in spo_list]}"
        # 删除长度过长、没有找到实体信息的句子
        if len(text) > keep_max_length or len(spo) > keep_max_length or len(spo_list) == 0:
            continue

        my_raw_data.append({
                'prompt': text, 
                'response':spo.replace('\'','').replace(' ', ''),
            })


    dev_date = []
    if dev_file_name is not None:
        dev_index = np.random.choice(range(0, len(my_raw_data)), size=dev_size, replace=False)
        dev_index = set(dev_index)
        assert len(dev_index) == dev_size
        
        train_data = [x for i, x in enumerate(my_raw_data) if i not in dev_index]
        dev_date = [x for i, x in enumerate(my_raw_data) if i in dev_index]
        
        with codecs.open(dev_file_name, 'w', encoding='utf-8') as f:
            ujson.dump(dev_date, f, indent=4, ensure_ascii=False)
        
        my_raw_data = train_data

    print(f'length of train data {len(my_raw_data)}, length of eval data {len(dev_date)}')
    
    with codecs.open(train_file_name, 'w', encoding='utf-8') as f:
        ujson.dump(my_raw_data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    raw_data_file = './data/train_data.json'
    train_file = './data/my_train.json'
    dev_file = './data/my_eval.json'

    process_all_50_schemas('./data/all_50_schemas', './data/my_schemas.txt')

    process_data(raw_data_file, train_file, dev_file, keep_max_length=512, dev_size=1000)

    # 使用该数据集公开的dev_data作为测试集
    process_data('./data/dev_data.json', train_file_name='./data/test.json', dev_file_name=None, keep_max_length=512, dev_size=1000)
