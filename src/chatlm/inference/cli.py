"""命令行交互式对话。

用法::

    chatlm-cli
    chatlm-cli --no-stream
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from threading import Event, Thread

from rich.live import Live
from rich.text import Text

from chatlm.config import InferConfig
from chatlm.inference.chatbot import ChatBot

EMPTY_ANSWER = "我是一个参数很少的AI模型🥺，知识库较少，无法直接回答您的问题，换个问题试试吧👋"

WELCOME_TXT = "欢迎使用ChatBot，输入`exit`退出，输入`cls`清屏。\n"

CLEAR_CMD = "cls" if platform.system().lower() == "windows" else "clear"


def build_prompt(history: list[list[str]], welcome_txt: str = WELCOME_TXT) -> str:
    """把历史对话拼接为待打印文本。"""
    prompt = welcome_txt
    for query, response in history:
        prompt += f"\n\033[0;33;40m用户：\033[0m{query}"
        prompt += f"\n\033[0;32;40mChatBot：\033[0m\n{response}\n"
    return prompt


def busy_indicator(stop_event: Event, total_time: int = 60) -> None:
    """非 stream chat 打印忙碌状态。"""
    list_circle = ["\\", "|", "/", "—"]
    for i in range(total_time * 4):
        time.sleep(0.25)
        print(f"\r{list_circle[i % 4]}", end="", flush=True)

        if stop_event.is_set():
            break

    print("\r", end="", flush=True)


def chat(chat_bot: ChatBot, stream: bool = True) -> None:
    """进入交互式对话循环。"""
    history: list[list[str]] = []
    turn_count = 0

    print(WELCOME_TXT)

    while True:
        print("\r\033[0;33;40m用户：\033[0m", end="", flush=True)
        input_txt = input()

        if len(input_txt) == 0:
            print("请输入问题")
            continue

        # 退出
        if input_txt.lower() == "exit":
            break

        # 清屏
        if input_txt.lower() == "cls":
            history = []
            turn_count = 0
            os.system(CLEAR_CMD)
            print(WELCOME_TXT)
            continue

        if not stream:
            stop_event = Event()
            thread = Thread(target=busy_indicator, args=(stop_event,), daemon=True)
            thread.start()

            outs = chat_bot.chat(input_txt)

            stop_event.set()
            thread.join()

            print(f"\r\033[0;32;40mChatBot：\033[0m\n{outs}\n\n", end="")

            continue

        history.append([input_txt, ""])
        stream_txt = []
        streamer = chat_bot.stream_chat(input_txt)
        rich_text = Text()

        print("\r\033[0;32;40mChatBot：\033[0m\n", end="")

        with Live(rich_text, refresh_per_second=15) as _live:
            for word in streamer:
                rich_text.append(word)
                stream_txt.append(word)

        stream_txt_str = "".join(stream_txt)

        if len(stream_txt_str) == 0:
            stream_txt_str = EMPTY_ANSWER

        history[turn_count][1] = stream_txt_str

        os.system(CLEAR_CMD)
        print(build_prompt(history), flush=True)
        turn_count += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatLM 命令行对话")
    parser.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    args = parser.parse_args()

    chat_bot = ChatBot(infer_config=InferConfig())
    chat(chat_bot, stream=not args.no_stream)


if __name__ == "__main__":
    main()


__all__ = ["build_prompt", "busy_indicator", "chat", "main"]
