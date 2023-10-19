from dataclasses import dataclass
from typing import Union

import uvicorn
from fastapi import FastAPI, Depends, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from model.infer import ChatBot
from config import InferConfig

CONFIG = InferConfig()
chat_bot = ChatBot(infer_config=CONFIG)

#==============================================================
# api 配置

# api根目录
ROOT = '/api'

# api key
USE_AUTH = False if len(CONFIG.api_key) == 0 else True
SECRET_KEY = CONFIG.api_key

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

#==============================================================

async def api_key_auth(token: str = Depends(oauth2_scheme)) -> Union[None, bool]:
  """
  验证post请求的key是否和服务器的key一致
  需要在请求头加上 Authorization: Bearer SECRET_KEY
  """
  if not USE_AUTH:
    return None  # return None if not auth

  if token == SECRET_KEY:
    return None # return None if auth success

  # 验证出错
  raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail="api认证未通过，请检查认证方式和token！",
      headers={"WWW-Authenticate": "Bearer"},
  )

# pos请求json
class ChatInput(BaseModel):
  input_txt: str


@app.post(ROOT + "/chat")
async def chat(post_data: ChatInput, authority: str = Depends(api_key_auth)) -> dict:
    """
    post 输入: {'input_txt': '输入的文本'}
    response: {'response': 'chatbot文本'}
    """
    input_txt = post_data.input_txt
    if len(input_txt) == 0:
        raise HTTPException(
                            status_code=status.HTTP_406_NOT_ACCEPTABLE,
                            detail="input_txt length = 0 is not allow!",
                            headers={"WWW-Authenticate": "Bearer"},
                        )
    
    outs = chat_bot.chat(input_txt)

    return {'response': outs}

if __name__ == '__main__':
  
  # 加上reload参数（reload=True）时，多进程设置无效
  # workers = max(multiprocessing.cpu_count() * CONFIG.getint('uvicorn','process_worker'), 1)
  workers = max(CONFIG.workers, 1)
  print('启动的进程个数:{}'.format(workers))

  uvicorn.run(
      'api_demo:app',
      host=CONFIG.host,
      port=CONFIG.port,
      reload=CONFIG.reload,
      workers=workers,
      log_level='info'
  )


# 服务方式启动：
# 命令行输入：uvicorn api_demo:app --host 0.0.0.0 --port 8094 --workers 8
# api_demo：api_demo.py文件
# app：app = FastAPI() 在main.py内创建的对象。
# --reload：在代码更改后重新启动服务器。 只有在开发时才使用这个参数，此时多进程设置会无效
