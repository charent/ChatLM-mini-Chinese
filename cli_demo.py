import platform
import os
import time
from threading import Thread

from rich.text import Text
from rich.live import Live

from model.infer import ChatBot
from config import InferConfig
from utils.functions import fixed_response, fixed_en

infer_config = InferConfig()
chat_bot = ChatBot(infer_config=infer_config)

clear_cmd = 'cls' if platform.system().lower() == 'windows' else 'clear'

welcome_txt = '欢迎使用ChatBot，输入`exit`或者`quit` 退出，输入`cls`或者`clear`清屏。\n'
print(welcome_txt)

def build_prompt(history: list[list[str]]) -> str:
    prompt = welcome_txt
    for query, response in history:
        prompt += '\nuser: {}'.format(query)
        prompt += '\nchat_bot: {}\n'.format(response)
    return prompt

STOP_CIRCLE: bool=False
def circle_print(total_time: int=60) -> None:
    global STOP_CIRCLE
    '''非stream chat打印忙碌状态
    '''
    list_circle = ["\\", "|", "/", "—"]
    for i in range(total_time * 4):
        time.sleep(0.25)
        print("\r{}".format(list_circle[i % 4]), end="", flush=True)

        if STOP_CIRCLE: break

    print("\r", end='', flush=True)


def chat(stream: bool=True) -> None:
    global  STOP_CIRCLE
    history = []
    turn_count = 0

    while True:
        print('\r\033[0;33;40m用户：\033[0m', end='', flush=True)
        input_txt = input()

        if len(input_txt) == 0:
            print('请输入问题')
            continue
        
        # 退出
        if input_txt.lower() in ('exit', 'quite'):
            break
        
        # 清屏
        if input_txt.lower() in ('cls', 'clear'):
            history = []
            turn_count = 0
            os.system(clear_cmd)
            print(welcome_txt)
            continue
        
        if not stream:
            STOP_CIRCLE = False
            thread = Thread(target=circle_print)
            thread.start()

            outs = chat_bot.chat(input_txt)

            STOP_CIRCLE = True
            thread.join()

            print("\r\033[0;32;40mChatBot：\033[0m\n{}\n\n".format(outs), end='')
           
            continue

        history.append([input_txt, ''])
        stream_txt = ''
        streamer = chat_bot.stream_chat(input_txt)
        rich_text = Text()

        with Live(rich_text, refresh_per_second=10) as live: 
            for i, word in enumerate(streamer):
                word = word.replace(' ', '')
                stream_txt += word
                rich_text.append(word)

        stream_txt = fixed_en(fixed_response(stream_txt))
        history[turn_count][1] = stream_txt
        
        os.system(clear_cmd)
        print(build_prompt(history), flush=True)
        turn_count += 1

if __name__ == '__main__':
    chat(stream=False)