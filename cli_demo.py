import platform
import os

from rich.text import Text
from rich.live import Live

from model.infer import ChatBot
from config import InferConfig

infer_config = InferConfig()
chat_bot = ChatBot(infer_config=infer_config)

clear_cmd = 'cls' if platform.system().lower() == 'windows' else 'clear'

welcome_txt = '欢迎使用ChatBot，输入`exit`、`quite` 退出.'
print(welcome_txt)

def build_prompt(history: list[list[str]]) -> str:
    prompt = welcome_txt
    for query, response in history:
        prompt += '\nuser: {}'.format(query)
        prompt += '\nchat_bot: {}\n'.format(response)
    return prompt


def chat(stream: bool=True) -> None:
    history = []
    turn_count = 0

    while True:
        input_txt = input('user: ')

        if len(input_txt) == 0:
            print('please input somthing')
        
        if input_txt.lower() in ('exit', 'quite'):
            break
        
        if not stream:
            outs = chat_bot.chat(input_txt)
            print("chat_bot:\n{}\n".format(outs))
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

        history[turn_count][1] = stream_txt[0: stream_txt.rfind('。') + 1]

        os.system(clear_cmd)
        print(build_prompt(history), flush=True)
        turn_count += 1

if __name__ == '__main__':
    chat(stream=False)