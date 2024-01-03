import torch
from torch import Tensor, LongTensor
from transformers import T5ForConditionalGeneration, T5Config
from transformers import TextIteratorStreamer
from transformers.generation.configuration_utils import GenerationConfig

class TextToTextModel(T5ForConditionalGeneration):
    def __init__(self, config: T5Config) -> None:
        '''
            TextToTextModel继承T5ForConditionalGeneration
        '''
        super().__init__(config)
    
    @torch.no_grad()
    def my_generate(self, 
                input_ids: LongTensor, 
                attention_mask: LongTensor, 
                max_seq_len: int=256,
                search_type: str='beam',
                streamer: TextIteratorStreamer=None,
            ) -> Tensor:
        '''
        自定义gennerate方法方便调用、测试
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
        generation_config.decoder_start_token_id = self.config.decoder_start_token_id
        generation_config.max_new_tokens = max_seq_len
        # generation_config.repetition_penalty = 1.1 # 重复词惩罚

        if search_type == 'greedy':
            generation_config.num_beams = 1
            generation_config.do_sample = False
        elif search_type == 'beam':
            generation_config.top_k = 50
            generation_config.num_beams = 5
            generation_config.do_sample = True
            generation_config.top_p = 0.95
            generation_config.no_repeat_ngram_size = 4
            generation_config.length_penalty = -2.0
            generation_config.early_stopping = True
        elif search_type == 'sampling':
            generation_config.num_beams = 1
            generation_config.do_sample = True
            generation_config.top_k = 50
            generation_config.temperature = 0.98   # 越低，贫富差距越大，越高(>1)，越趋向于均匀分布
            generation_config.top_p = 0.80
            generation_config.no_repeat_ngram_size = 4
        elif search_type == 'contrastive':
            generation_config.penalty_alpha = 0.5
            generation_config.top_k = 50

        result = self.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            streamer=streamer,
            )

        return result
