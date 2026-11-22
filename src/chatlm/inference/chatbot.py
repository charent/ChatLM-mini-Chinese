import os
from threading import Thread
from typing import cast

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from safetensors.torch import load_model
from transformers import PreTrainedTokenizerFast, TextIteratorStreamer

# import 自定义类和函数
from chatlm.config import InferConfig, T5ModelConfig
from chatlm.models.chat_model import TextToTextModel
from chatlm.utils.hf import get_T5_config


class ChatBot:
    def __init__(self, infer_config: InferConfig) -> None:
        """ """
        self.infer_config = infer_config
        self.model: TextToTextModel
        # 初始化tokenizer
        tokenizer = PreTrainedTokenizerFast.from_pretrained(infer_config.model_dir)
        self.tokenizer = tokenizer
        self.encode = tokenizer.encode_plus
        self.batch_decode = tokenizer.batch_decode
        self.batch_encode_plus = tokenizer.batch_encode_plus

        t5_config = get_T5_config(
            T5ModelConfig(), vocab_size=len(tokenizer), decoder_start_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
        )

        try:
            model = TextToTextModel(t5_config)

            if os.path.isdir(infer_config.model_dir):
                # from_pretrained
                model = cast(TextToTextModel, TextToTextModel.from_pretrained(infer_config.model_dir))

            elif infer_config.model_dir.endswith(".safetensors"):
                # load safetensors
                load_model(model, infer_config.model_dir)

            else:
                # load torch checkpoint
                model.load_state_dict(torch.load(infer_config.model_dir))

            self.model = model

        except Exception as e:
            print(str(e), "transformers and pytorch load fail, try accelerate load function.")

            empty_model = None
            with init_empty_weights():
                empty_model = TextToTextModel(t5_config)

            self.model = cast(
                TextToTextModel,
                load_checkpoint_and_dispatch(
                    model=empty_model,
                    checkpoint=infer_config.model_dir,
                    device_map="auto",
                    dtype=torch.float16,
                ),
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # transformers / accelerate 存根对 from_pretrained 返回值的推断不稳定，这里按 nn.Module 接口调用
        self.model.to(self.device)  # pyright: ignore[reportArgumentType]

        self.streamer = TextIteratorStreamer(tokenizer=tokenizer, clean_up_tokenization_spaces=True, skip_special_tokens=True)

    def stream_chat(self, input_txt: str) -> TextIteratorStreamer:
        """
        流式对话，线程启动后可返回，通过迭代streamer获取生成的文字，仅支持greedy search
        """
        encoded = self.encode(input_txt + "[EOS]")

        input_ids = torch.LongTensor([encoded.input_ids]).to(self.device)
        attention_mask = torch.LongTensor([encoded.attention_mask]).to(self.device)

        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_seq_len": self.infer_config.max_seq_len,
            "streamer": self.streamer,
            "search_type": "greedy",
        }

        thread = Thread(target=self.model.my_generate, kwargs=generation_kwargs)
        thread.start()

        return self.streamer

    def chat(self, input_txt: str | list[str]) -> str | list[str]:
        """
        非流式生成，可以使用beam search、beam sample等方法生成文本。
        """
        if isinstance(input_txt, str):
            input_txt = [input_txt]
        elif not isinstance(input_txt, list):
            raise Exception("input_txt mast be a str or list[str]")

        # add EOS token
        input_txts = [f"{txt}[EOS]" for txt in input_txt]
        encoded = self.batch_encode_plus(input_txts, padding=True)
        input_ids = torch.LongTensor(encoded.input_ids).to(self.device)
        attention_mask = torch.LongTensor(encoded.attention_mask).to(self.device)

        outputs = self.model.my_generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_seq_len=self.infer_config.max_seq_len,
            search_type="greedy",
        )

        outputs = self.batch_decode(outputs.cpu().numpy(), clean_up_tokenization_spaces=True, skip_special_tokens=True)

        note = "我是一个参数很少的AI模型🥺，知识库较少，无法直接回答您的问题，换个问题试试吧👋"
        outputs = [item if len(item) != 0 else note for item in outputs]

        return outputs[0] if len(outputs) == 1 else outputs
