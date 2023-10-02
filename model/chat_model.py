from os.path import dirname, abspath
import sys
import torch
from torch.nn import Module
from torch import Tensor, LongTensor
from transformers import T5ForConditionalGeneration, T5Config

sys.path.append('.')
sys.path.append('..')

from config import T5ModelConfig
from config import PROJECT_ROOT

class TextToTextModel(Module):
    def __init__(self, config: T5ModelConfig, decoder_start_token_id: int=0) -> None:
        '''
        默认T5Config {
            "d_ff": 2048,                   # 全连接层维度
            "d_kv": 64,                     # 注意力头数, d_model // num_heads == d_kv
            "d_model": 512,                 # 词向量维度
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
            "vocab_size": 20480             # 词库大小
        }
        '''
        super(TextToTextModel, self).__init__()

        assert config.max_seq_len > 0
        assert config.d_model // config.num_heads == config.d_kv

        t5_config = T5Config()

        #  "max_seq_len": 128,
        # "d_ff": 128,                   # 全连接层维度
        # "d_kv": 16,                     # 注意力头数, d_model // num_heads == d_kv
        # "d_model": 128,                 # 词向量大学
        # "num_decoder_layers": 2,        # Transformer decoder 隐藏层层数
        # "num_heads": 8,
        # "num_layers": 2,                #  Transformer encoder 隐藏层层数
        # "vocab_size": 4096             # 词库大大小
        
        # 初始化
        t5_config.d_ff = config.d_ff
        t5_config.d_kv = config.d_kv
        t5_config.d_model = config.d_model
        t5_config.num_decoder_layers = config.num_decoder_layers
        t5_config.num_heads = config.num_heads
        t5_config.num_layers = config.num_layers
        t5_config.vocab_size = config.vocab_size
        t5_config.decoder_start_token_id = decoder_start_token_id
        # print(t5_config)

        self.user_config = config
        self.t5_config = t5_config
        
        self.model = T5ForConditionalGeneration(t5_config)

    def forward(self, input_ids: LongTensor, input_mask: LongTensor, labels: LongTensor, **args) -> Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=input_mask,
            labels=labels,
            **args
            )

    def generate(self, input_ids: LongTensor, attention_mask: LongTensor) -> Tensor:
        result = self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            max_length=self.user_config.max_seq_len, 
            do_sample=True, 
            # top_p=0.6,
            top_k=50,
            early_stopping=True,
            # num_beams=2,
            # repetition_penalty=2.5,
            # length_penalty=1.0,
            )

        return result

if __name__ == '__main__':
    model_dir = PROJECT_ROOT + '/model_save/t2t/'

    config = T5ModelConfig()

    config.max_seq_len = 128
    config.d_ff = 128
    config.d_kv = 16
    config.d_model = 128
    config.num_decoder_layers = 2
    config.num_heads = 8
    config.num_layers = 2
    config.vocab_size = 4096
    config.decoder_start_token_id = 0  # In T5 it is usually set to the pad_token_id. See T5 docs for more information

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TextToTextModel(config)
    model = model.to(device)

    size = (8, 128)

    inputs = torch.randint(low=0, high=2048, size=size).to(device)
    mask = torch.randint(low=0, high=2, size=size).to(device)
    target_ids = torch.randint(low=0, high=2048, size=size).to(device)

    outs = model(input_ids=inputs, input_mask=mask, labels=target_ids)
    
    # Seq2SeqLMOutput
    # loss=loss,
    # logits=lm_logits,
    # past_key_values=decoder_outputs.past_key_values,
    # decoder_hidden_states=decoder_outputs.hidden_states,
    # decoder_attentions=decoder_outputs.attentions,
    # cross_attentions=decoder_outputs.cross_attentions,
    # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
    # encoder_hidden_states=encoder_outputs.hidden_states,
    # encoder_attentions=encoder_outputs.attentions,
    
    print("device:{}, inputs:{}, mask:{}, target_ids:{}, outs.logits:{}, outs.loos:{}".format(
        device,
        inputs.shape,
        mask.shape,
        target_ids.shape,
        outs.logits.shape,
        outs.loss.shape
    ))

    outs = model.generate(input_ids=inputs, attention_mask=mask)
    print('generate.shape:', outs.shape)

    # print(outs)