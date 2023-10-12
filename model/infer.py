from threading import Thread
import platform

import torch

from transformers import TextIteratorStreamer

from tokenizers import Tokenizer
from accelerate import init_empty_weights, dispatch_model,load_checkpoint_in_model,load_checkpoint_and_dispatch
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

# import 自定义类和函数
from model.chat_model import TextToTextModel
from utils.logger import Logger
from utils.functions import json_to_dataclass
from config import InferConfig

class ChatBot:
    def __init__(self, infer_config: InferConfig) -> None:
        '''
        '''
        
        self.infer_config = infer_config
        
        model_config_class = json_to_dataclass(infer_config.model_config_file, 'ModelConfig')
        self.model_config = model_config_class()

        # file_name=None会自动生成以当前日期命名的log文件名
        self.logger = Logger('chat_logs', std_out=True, save2file=True, file_name=None)

         # 初始化tokenizer
        tokenizer = Tokenizer.from_file(infer_config.tokenizer_file)
        tokenizer.enable_padding(length=infer_config.max_seq_len)
        tokenizer.enable_truncation(max_length=infer_config.max_seq_len)
        self.tokenizer = tokenizer
        self.encode = tokenizer.encode
        self.decode_batch = tokenizer.decode_batch

        empty_model = None
        with init_empty_weights():
            empty_model = TextToTextModel(config=self.model_config, decoder_start_token_id=tokenizer.token_to_id('[PAD]'))

        if torch.cuda.device_count() >= 2 and platform.system().lower() == 'linux':
            # 量化配置
            bnb_quantization_config = BnbQuantizationConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, 
                bnb_4bit_quant_type="nf4"
                )
            
            # 加载模型
            model = load_and_quantize_model(
                    model=empty_model, 
                    weights_location=infer_config.model_file, 
                    bnb_quantization_config=bnb_quantization_config, 
                )
            
            # 多GPU分发
            self.model = dispatch_model(
                model=model,
                device_map='auto'
            )   
                
        else:
            self.model = load_checkpoint_and_dispatch(
                model=empty_model,
                checkpoint=infer_config.model_file,
                device_map='auto',
                dtype=torch.float16,
            )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.steamer = TextIteratorStreamer(tokenizer=tokenizer)

    def stream_chat(self, input_txt: str) -> TextIteratorStreamer:
        encoded = self.encode(input_txt)
        
        input_ids = torch.LongTensor([encoded.ids]).to(self.device)
        attention_mask = torch.LongTensor([encoded.attention_mask]).to(self.device)

        generation_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_seq_len': self.infer_config.max_seq_len,
            'streamer': self.steamer,
        }

        thread = Thread(target=self.model.steam_generate, kwargs=generation_kwargs)
        thread.start()
        
        return self.steamer

    
    def chat(self, input_txt: str, ) -> str:
        '''
        '''
        encoded = self.encode(input_txt)
        
        input_ids = torch.LongTensor([encoded.ids]).to(self.device)
        attention_mask = torch.LongTensor([encoded.attention_mask]).to(self.device)

        outputs = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_seq_len=self.infer_config.max_seq_len,
                        )

        outputs = self.decode_batch(outputs.cpu().numpy(),  skip_special_tokens=True)

        # 删除decode出来字符间的空格
        outputs = [sentance.replace(' ', '') for sentance in outputs][0]
        outputs = outputs[0: outputs.rfind('。') + 1]

        return outputs
