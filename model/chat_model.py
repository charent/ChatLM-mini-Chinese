from typing import Union

import torch
from torch.nn import Module
from torch import Tensor, LongTensor
from transformers import T5ForConditionalGeneration, T5Config
from transformers import TextIteratorStreamer
from transformers.generation.configuration_utils import GenerationConfig

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

    def forward(self, input_ids: LongTensor, input_mask: LongTensor, labels: LongTensor, decoder_attention_mask: LongTensor=None,  **args) -> Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=input_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            **args
            )

    
    @torch.no_grad()
    def my_generate(self, 
                input_ids: LongTensor, 
                attention_mask: LongTensor, 
                max_seq_len: int=256,
                search_type: str='beam',
            ) -> Tensor:
        '''
        search_type: ['greedy', 'beam', 'sampling', 'contrastive', ]

        - *greedy decoding* by calling [`~generation.GenerationMixin.greedy_search`] if `num_beams=1` and
            `do_sample=False`
        - *contrastive search* by calling [`~generation.GenerationMixin.contrastive_search`] if `penalty_alpha>0.`
            and `top_k>1`
        - *multinomial sampling* by calling [`~generation.GenerationMixin.sample`] if `num_beams=1` and
            `do_sample=True`
        - *beam-search decoding* by calling [`~generation.GenerationMixin.beam_search`] if `num_beams>1` and
            `do_sample=False`
        - *beam-search multinomial sampling* by calling [`~generation.GenerationMixin.beam_sample`] if
            `num_beams>1` and `do_sample=True`
        '''
        generation_config = GenerationConfig()
        generation_config.remove_invalid_values = True
        generation_config.eos_token_id = 1
        generation_config.pad_token_id = 0
        generation_config.decoder_start_token_id = self.t5_config.decoder_start_token_id
        generation_config.max_new_tokens = max_seq_len
        generation_config.repetition_penalty = 1.5 # 重复词惩罚


        if search_type == 'greedy':
            generation_config.num_beams = 1
            generation_config.do_sample = False
        elif search_type == 'beam':
            generation_config.top_k = 50
            generation_config.num_beams = 5
            generation_config.do_sample = True
            generation_config.top_p = 0.95
            generation_config.no_repeat_ngram_size = 4
            generation_config.early_stopping = True
        elif search_type == 'sampling':
            generation_config.num_beams = 1
            generation_config.do_sample = True
            generation_config.top_k = 50
            generation_config.temperature = 0.98   # 越低概率越趋向于均匀分布
            generation_config.top_p = 0.80
            generation_config.no_repeat_ngram_size = 4
        elif search_type == 'contrastive':
            generation_config.penalty_alpha = 0.5
            generation_config.top_k = 50

        result = self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,

            )

        return result

    @torch.no_grad()
    def generate(self, 
                input_ids: LongTensor, 
                attention_mask: LongTensor, 
                max_seq_len: int=256,
            ) -> Tensor:
        
        result = self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            no_repeat_ngram_size=4,
            
            # top_k args:
            # max_length=max_seq_len, 
            do_sample=True, 
            top_k=50,
            early_stopping=True,
            num_beams=5,
            remove_invalid_values=True,
            temperature=0.99,
            # top_p args:
            top_p=0.10,
            max_new_tokens=max_seq_len,
            decoder_start_token_id=self.t5_config.decoder_start_token_id,
            repetition_penalty=1.5, # 重复词惩罚
            # length_penalty=2.5,  # <0鼓励生成长句子
            eos_token_id=1,
            pad_token_id=0,
            )

        return result
    
    @torch.no_grad()
    def stream_generate(self,
                    input_ids: LongTensor, 
                    attention_mask: LongTensor, 
                    max_seq_len: int=256,
                    streamer: TextIteratorStreamer=None,
                ) -> None:
        
        self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            streamer=streamer,
            max_length=max_seq_len, 
            
        )
    

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