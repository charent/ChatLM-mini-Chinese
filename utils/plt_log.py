# log 画图
from datetime import datetime
import numpy as np
import pandas as pd 

from matplotlib import pyplot as plt

from config import PROJECT_ROOT

def str_to_timestamp(string: str) -> float:
    '''
    '''
    date_fmt = '%Y-%m-%d %H:%M:%S.%f'
    string = string.replace('[', '').replace(']', '')

    # 转化为时间戳
    return datetime.strptime(string, date_fmt).timestamp()

def plot_traing_loss(log_file: str, start_date: str, end_date: str, pic_save_to_file: str=None) -> None:
    '''
    将log日志中记录的画图，按需保存到文件，由于log日志打印内容较多，需要指定要打印loss的开始时间和结束时间
    examlpe:
    >>>  plot_traing_loss('./logs/trainer.log', '[2023-10-01 08:44:39.303]', '[2023-10-01 11:29:12.376]')
    >>> plot_traing_loss('./logs/trainer.log', '2023-10-01 08:44:39.303', '2023-10-01 11:29:12.376')
    '''
    start_timestamp = str_to_timestamp(start_date)
    end_timestamp = str_to_timestamp(end_date)
    
    loss_list = []
    with open(log_file, 'r', encoding='utf-8') as f:

        for line in f:
            if 'training loss: epoch:' in line:
                line = line.split(' ')
                date = ' '.join(line[0: 2])
                if str_to_timestamp(date) < start_timestamp:
                    continue
                
                if str_to_timestamp(date) > end_timestamp:
                    break

                if len(line) != 8: continue

                epoch = line[5][6: -1]  # 'epoch:0,'
                step = line[6][5: -1]   # 'step:0,'
                loss = float(line[7][5: -1])   # 'loss:0.11086619377136231\n'
                loss_list.append([epoch, step, loss])
    
    df = pd.DataFrame(loss_list, columns=['epoch', 'step', 'loss'])

    plt.figure()                 
    plt.plot(df['loss'],'b',label = 'loss')
    plt.ylabel('loss')
    plt.xlabel('step')
    plt.legend()        #个性化图例（颜色、形状等）
    plt.show()

    if not pic_save_to_file:
        plt.savefig(pic_save_to_file) 


if __name__ == '__main__':
    
    plot_traing_loss(PROJECT_ROOT + '/logs/trainer.log', '[2023-10-01 08:44:39.303]', '[2023-10-01 11:29:12.376]')