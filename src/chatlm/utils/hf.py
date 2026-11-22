"""HuggingFace 相关的配置转换与训练回调。"""

from __future__ import annotations

import torch
from transformers import T5Config, TrainerCallback, TrainingArguments
from transformers.trainer_callback import TrainerControl, TrainerState

from chatlm.config import T5ModelConfig


class MyTrainerCallback(TrainerCallback):
    """低显存设备的辅助回调。"""

    log_cnt = 0

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        """在打印 n 次日志后清除 cuda 缓存，适合低显存设备，能防止 OOM。"""
        self.log_cnt += 1
        if self.log_cnt % 2 == 0:
            torch.cuda.empty_cache()

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """在 on_epoch_end 时保存一次模型。

        TrainingArguments 的 save_strategy 中 epoch 和 steps 不兼容。要实现每隔
        save_steps 步保存一次检查点，考虑到磁盘空间大小，最多只保存最近 N 个检查点。
        """
        # 设置should_save=True并返回即可
        control.should_save = True
        return control


def get_T5_config(
    config: T5ModelConfig,
    vocab_size: int,
    decoder_start_token_id: int = 0,
    eos_token_id: int = 1,
) -> T5Config:
    """用户配置转换为 T5Config。"""
    t5_config = T5Config()
    # 初始化
    t5_config.d_ff = config.d_ff
    t5_config.d_kv = config.d_kv
    t5_config.d_model = config.d_model
    t5_config.num_decoder_layers = config.num_decoder_layers
    t5_config.num_heads = config.num_heads
    t5_config.num_layers = config.num_layers
    t5_config.vocab_size = vocab_size
    t5_config.decoder_start_token_id = decoder_start_token_id
    t5_config.eos_token_id = eos_token_id

    return t5_config


__all__ = ["MyTrainerCallback", "get_T5_config"]
