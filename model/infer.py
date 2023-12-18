from threading import Thread
import platform
from typing import Union
import torch

from transformers import TextIteratorStreamer,PreTrainedTokenizerFast
from safetensors.torch import load_model

from accelerate import init_empty_weights, dispatch_model,load_checkpoint_in_model, load_checkpoint_and_dispatch
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

# import 自定义类和函数
from model.chat_model import TextToTextModel
from utils.functions import json_to_dataclass, fixed_space

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
        tokenizer = PreTrainedTokenizerFast.from_pretrained(infer_config.tokenizer_dir)
        self.tokenizer = tokenizer
        self.encode = tokenizer.encode_plus
        self.batch_decode = tokenizer.batch_decode
        self.batch_encode_plus = tokenizer.batch_encode_plus
        
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
                # print(str(e), '`accelerate` load fail, try another load function.')
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
        '''
        流式对话，线程启动后可返回，通过迭代streamer获取生成的文字，仅支持greedy search
        '''
        encoded = self.encode(input_txt + '[EOS]')
        
        input_ids = torch.LongTensor([encoded.input_ids]).to(self.device)
        attention_mask = torch.LongTensor([encoded.attention_mask]).to(self.device)

        generation_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_seq_len': self.infer_config.max_seq_len,
            'streamer': self.streamer,
            'search_type': 'greedy',
        }

        thread = Thread(target=self.model.my_generate, kwargs=generation_kwargs)
        thread.start()
        
        return self.streamer
    
    def chat(self, input_txt: Union[str, list[str]] ) -> Union[str, list[str]]:
        '''
        非流式生成，可以使用beam search、beam sample等方法生成文本。
        '''
        if isinstance(input_txt, str):
            input_txt = [input_txt]
        elif not isinstance(input_txt, list):
            raise Exception('input_txt mast be a str or list[str]')
        
        # add EOS token
        input_txts = [f"{txt}[EOS]" for txt in input_txt]
        encoded = self.batch_encode_plus(input_txts,  padding=True)
        input_ids = torch.LongTensor(encoded.input_ids).to(self.device)
        attention_mask = torch.LongTensor(encoded.attention_mask).to(self.device)

        outputs = self.model.my_generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_seq_len=self.infer_config.max_seq_len,
                            search_type='greedy',
                        )

        outputs = self.batch_decode(outputs.cpu().numpy(),  clean_up_tokenization_spaces=True, skip_special_tokens=True)

        note = "我是一个参数很少的AI模型🥺，知识库较少，无法直接回答您的问题，换个问题试试吧👋"
        outputs = [item if len(item) != 0 else note for item in outputs]

        # 删除decode出来字符间的空格
        outputs = [fixed_space(item) for item in outputs]

        return outputs[0] if len(outputs) == 1 else outputs
