"""FastAPI 推理服务。

启动方式::

    chatlm-api
    # 或
    uvicorn chatlm.inference.api:app --host 0.0.0.0 --port 8812

post 请求地址: http://127.0.0.1:8812/api/chat
需要添加 Authorization 头（配置了 api_key 时），body json 格式::

    {"input_txt": "感冒了要怎么办"}
"""

from __future__ import annotations

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from chatlm.config import InferConfig
from chatlm.inference.chatbot import ChatBot

CONFIG = InferConfig()
chat_bot = ChatBot(infer_config=CONFIG)

# ==============================================================
# api 配置

# api 根目录
ROOT = "/api"

# api key
USE_AUTH = bool(CONFIG.api_key)
SECRET_KEY = CONFIG.api_key

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")


async def api_key_auth(token: str = Depends(oauth2_scheme)) -> None:
    """验证 post 请求的 key 是否和服务器的 key 一致。

    需要在请求头加上 ``Authorization: Bearer <SECRET_KEY>``。
    """
    if not USE_AUTH:
        return None  # return None if not auth

    if token == SECRET_KEY:
        return None  # return None if auth success

    # 验证出错
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="api认证未通过，请检查认证方式和token！",
        headers={"WWW-Authenticate": "Bearer"},
    )


class ChatInput(BaseModel):
    """post 请求 json。"""

    input_txt: str


@app.post(ROOT + "/chat")
async def chat(post_data: ChatInput, authority: str = Depends(api_key_auth)) -> dict:
    """post 输入: {'input_txt': '输入的文本'}；response: {'response': 'chatbot文本'}。"""
    input_txt = post_data.input_txt
    if len(input_txt) == 0:
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail="input_txt length = 0 is not allow!",
            headers={"WWW-Authenticate": "Bearer"},
        )

    outs = chat_bot.chat(input_txt)

    if len(outs) == 0:
        outs = "我是一个参数很少的AI模型🥺，知识库较少，无法直接回答您的问题，换个问题试试吧👋"

    return {"response": outs}


def main() -> None:
    """启动 uvicorn 服务。"""
    # 加上reload参数（reload=True）时，多进程设置无效
    workers = max(CONFIG.workers, 1)
    print(f"启动的进程个数:{workers}")

    uvicorn.run(
        "chatlm.inference.api:app",
        host=CONFIG.host,
        port=CONFIG.port,
        reload=CONFIG.reload,
        workers=workers,
        log_level=CONFIG.log_level,
    )


if __name__ == "__main__":
    main()


__all__ = ["app", "chat", "chat_bot", "main"]
