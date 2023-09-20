from os.path import dirname, abspath
import torch
from torch.nn import Module
from torch import Tensor
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config

ROOT_PATH = abspath(dirname(dirname(__file__)))

class TextToTextModel(Module):
    def __init__(self, config: dict) -> None:
        '''
        默认T5Config {
            "d_ff": 2048,                   # 全连接层维度
            "d_kv": 64,                     # 注意力头数, d_model // num_heads == d_kv
            "d_model": 512,                 # 词向量大学
            "dense_act_fn": "relu",
            "dropout_rate": 0.1,
            "eos_token_id": 1,
            "feed_forward_proj": "relu",
            "initializer_factor": 1.0,
            "is_encoder_decoder": true,
            "is_gated_act": false,
            "layer_norm_epsilon": 1e-06,
            "model_type": "t5",
            "num_decoder_layers": 6,        # Transformer decoder 隐藏层层数
            "num_heads": 8,
            "num_layers": 6,                #  Transformer encoder 隐藏层层数
            "pad_token_id": 0,
            "relative_attention_max_distance": 128,
            "relative_attention_num_buckets": 32,
            "transformers_version": "4.25.1",
            "use_cache": true,
            "vocab_size": 32128             # 词库大大小
        }
        '''
        super(TextToTextModel, self).__init__()

        t5_config = T5Config().__dict__
        print(t5_config)
        for k, v in config.items():
            if k in t5_config:
                t5_config[k] = v

        assert t5_config['d_model'] // t5_config['num_heads'] == t5_config['d_kv']
        
        self.model = T5ForConditionalGeneration(t5_config)

    def forward(self, **args) -> Tensor:
        return self.model(**args)




if __name__ == '__main__':
    model_dir = ROOT_PATH + '/model_save/t5-base/'

    config = {
        "d_ff": 128,                   # 全连接层维度
        "d_kv": 32,                     # 注意力头数, d_model // num_heads == d_kv
        "d_model": 128,                 # 词向量大学
        "num_decoder_layers": 2,        # Transformer decoder 隐藏层层数
        "num_heads": 7,
        "num_layers": 2,                #  Transformer encoder 隐藏层层数
        "vocab_size": 4096             # 词库大大小
    }
    model = TextToTextModel(config)
    inputs = torch.randint((8, 16, 32))
    print(inputs)