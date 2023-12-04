from threading import Thread
import platform

import torch

from transformers import TextIteratorStreamer,PreTrainedTokenizerFast
from safetensors.torch import load_model

from tokenizers import Tokenizer
from accelerate import init_empty_weights, dispatch_model,load_checkpoint_in_model,load_checkpoint_and_dispatch
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

# import 自定义类和函数
from model.chat_model import TextToTextModel
from utils.logger import Logger
from utils.functions import json_to_dataclass, fixed_response

from config import InferConfig

class ChatBot:
    def __init__(self, infer_config: InferConfig) -> None:
        '''
        '''
        
        self.infer_config = infer_config
        
        model_config_class = json_to_dataclass(infer_config.model_config_file, 'ModelConfig')
        self.model_config = model_config_class()

        # file_name=None会自动生成以当前日期命名的log文件名
        # self.logger = Logger('chat_logs', std_out=True, save2file=True, file_name=None)

         # 初始化tokenizer
        # tokenizer = Tokenizer.from_file(infer_config.tokenizer_file)
        # tokenizer.enable_padding(length=infer_config.max_seq_len)
        # tokenizer.enable_truncation(max_length=infer_config.max_seq_len)
        # self.tokenizer = tokenizer
        # self.encode = tokenizer.encode
        # self.decode_batch = tokenizer.decode_batch

        tokenizer_obj = Tokenizer.from_file(infer_config.tokenizer_file)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
        tokenizer.pad_token = '[PAD]'
        tokenizer.pad_token_id = tokenizer_obj.token_to_id('[PAD]')
        tokenizer.unk_token = '[UNK]'
        tokenizer.unk_token_id = tokenizer_obj.token_to_id('[UNK]')
        tokenizer.eos_token = '[EOS]'
        tokenizer.eos_token_id = tokenizer_obj.token_to_id('[EOS]')

        self.tokenizer = tokenizer
        self.encode = tokenizer.encode_plus
        self.batch_decode = tokenizer.batch_decode
        
        empty_model = None
        with init_empty_weights():
            empty_model = TextToTextModel(config=self.model_config, decoder_start_token_id=tokenizer.pad_token_id)

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
            try:
                self.model = load_checkpoint_and_dispatch(
                    model=empty_model,
                    checkpoint=infer_config.model_file,
                    device_map='auto',
                    dtype=torch.float16,
                )
            except Exception as e:
                print(str(e), '`accelerate` load fail, try another load function.')
                model = TextToTextModel(config=self.model_config, decoder_start_token_id=tokenizer.pad_token_id)

                if  infer_config.model_file.endswith('.safetensors'):
                    # load safetensors
                    load_model(model.model, infer_config.model_file) 
                else:
                    # load torch checkpoint
                    model.load_state_dict(torch.load(infer_config.model_file))  
                self.model = model

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.streamer = TextIteratorStreamer(tokenizer=tokenizer, clean_up_tokenization_spaces=True, skip_special_tokens=True)

    def stream_chat(self, input_txt: str) -> TextIteratorStreamer:
        encoded = self.encode(input_txt)
        
        input_ids = torch.LongTensor([encoded.input_ids]).to(self.device)
        attention_mask = torch.LongTensor([encoded.attention_mask]).to(self.device)

        generation_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_seq_len': self.infer_config.max_seq_len,
            'streamer': self.streamer,
        }

        thread = Thread(target=self.model.stream_generate, kwargs=generation_kwargs)
        thread.start()
        
        return self.streamer

    @staticmethod
    def fixed_en(stentance: str)->str:
        '''恢复被删除的英文空格
        '''
        n = len(stentance)
        new_sentance = []
        for i in range(0, n):
            if stentance[i].isupper() and i - 1 >= 0 and stentance[i - 1].islower() :
                new_sentance.append(' ')
            new_sentance.append(stentance[i])
            
        return ''.join(new_sentance)
    
    def chat(self, input_txt: str, ) -> str:
        '''
        '''
        encoded = self.encode(input_txt)
        
        input_ids = torch.LongTensor([encoded.input_ids]).to(self.device)
        attention_mask = torch.LongTensor([encoded.attention_mask]).to(self.device)

        outputs = self.model.my_generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_seq_len=self.infer_config.max_seq_len,
                            search_type='beam',
                        )

        outputs = self.batch_decode(outputs.cpu().numpy(),  clean_up_tokenization_spaces=True, skip_special_tokens=True)

        # 删除decode出来字符间的空格
        outputs = [sentance.replace(' ', '') for sentance in outputs][0]
        outputs = fixed_response(outputs)
        outputs = self.fixed_en(outputs)

        return outputs
