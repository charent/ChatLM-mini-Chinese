"""项目训练/推理配置。

所有路径默认值均来自 :mod:`chatlm.paths`，可用 ``field(default_factory=...)``
在实例化时求值，从而支持通过环境变量覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chatlm.paths import (
    ARTIFACT_DIR,
    DATA_DIR,
    DPO_DIR,
    LOG_DIR,
    LORA_DIR,
    PRETRAIN_DIR,
    SFT_DIR,
    TRAIN_STATE_DIR,
)
from chatlm.paths import (
    PROJECT_ROOT as _PROJECT_ROOT,
)

# 向后兼容：部分 notebook 仍以字符串形式使用项目根目录
PROJECT_ROOT: str = str(_PROJECT_ROOT)


# ===================================================================================
# 以下为推断的配置
@dataclass
class InferConfig:
    max_seq_len: int = 320  # 回答的最大长度
    mixed_precision: str = "bf16"  # 混合精度 ''no','fp16','bf16' or 'fp8'

    # 全量DPO模型文件, tokenizer文件和model权重放在同一个文件夹
    model_dir: str = field(default_factory=lambda: str(ARTIFACT_DIR))

    # this confing for api demo:
    api_key: str = ""
    host: str = "127.0.0.1"
    port: int = 8812
    reload: bool = True
    workers: int = 1
    log_level: str = "info"


# ===================================================================================
# 以下为dpo训练配置
@dataclass
class DpoConfig:
    max_seq_len: int = 512 + 8  # 8 for eos token
    sft_model_file: str = field(default_factory=lambda: str(ARTIFACT_DIR))

    tokenizer_dir: str = field(default_factory=lambda: str(ARTIFACT_DIR))  # tokenizer一般和model权重放在同一个文件夹

    dpo_train_file: str = field(default_factory=lambda: str(DATA_DIR / "my_dpo_data.json"))
    dpo_eval_file: str = field(default_factory=lambda: str(DATA_DIR / "my_dpo_eval.json"))

    adapter_file: str = field(default_factory=lambda: str(LORA_DIR / "adapter_model.safetensors"))
    log_dir: str = field(default_factory=lambda: str(LOG_DIR))

    per_device_train_batch_size: int = 4
    num_train_epochs: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    logging_first_step: bool = True
    logging_steps: int = 20
    save_steps: int = 2000
    output_dir: str = field(default_factory=lambda: str(DPO_DIR))
    warmup_steps: int = 1000
    fp16: bool = True
    seed: int = 23333
    beta: float = 0.1


# ===================================================================================
# 以下为sft配置
@dataclass
class SFTConfig:
    max_seq_len: int = 384 + 8  # 8 for eos token

    finetune_from_ckp_file: str = field(default_factory=lambda: str(ARTIFACT_DIR))

    tokenizer_dir: str = field(default_factory=lambda: str(ARTIFACT_DIR))  # tokenizer一般和model权重放在同一个文件夹
    sft_train_file: str = field(default_factory=lambda: str(DATA_DIR / "sft_train.json"))

    batch_size: int = 12
    num_train_epochs: int = 4
    save_steps: int = 5000
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    logging_first_step: bool = True
    logging_steps: int = 100
    output_dir: str = field(default_factory=lambda: str(SFT_DIR))
    warmup_steps: int = 100
    fp16: bool = True
    seed: int = 23333


# ===================================================================================
# 以下为训练的配置
@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size_per_gpu: int = 16

    learn_rate: float = 0.0001  # 最大 div_factor * learn_rate
    div_factor: int = 50

    mixed_precision: str = "bf16"  # 混合精度 ''no','fp16','bf16' or 'fp8'

    # 注意：计算梯度时相当于batch_size * gradient_accumulation_steps，说人话就是梯度累积步数>1时，等于增大n倍的batch_size
    gradient_accumulation_steps: int = 8  # 累积梯度更新步数

    warmup_steps: int = 1024  # 模型参数预热步数，预热样本数=warmup_steps * batch_size * gradient_accumulation_steps

    tokenizer_dir: str = field(default_factory=lambda: str(ARTIFACT_DIR))  # tokenizer一般和model权重放在同一个文件夹
    model_file: str = field(default_factory=lambda: str(ARTIFACT_DIR / "chat_small_t5.{}.bin"))
    model_config_file: str = field(default_factory=lambda: str(ARTIFACT_DIR / "model_config.json"))
    train_file: str = field(default_factory=lambda: str(DATA_DIR / "my_train_dataset_3k.parquet"))
    validation_file: str = field(default_factory=lambda: str(DATA_DIR / "my_valid_dataset_1k.parquet"))
    test_file: str = field(default_factory=lambda: str(DATA_DIR / "my_test_dataset_2k.parquet"))

    # 从哪个模型开始微调，仅当traing 函数 is_finetune = True时生效
    # 微调记得冻结某些层或者调低学习率
    finetune_from_ckp_file: str = field(default_factory=lambda: str(ARTIFACT_DIR / "chat_small_t5.best.bin"))

    # 训练状态保存，中断后可以从此处继续训练
    train_state_dir: str = field(default_factory=lambda: str(TRAIN_STATE_DIR))
    output_dir: str = field(default_factory=lambda: str(PRETRAIN_DIR))

    logging_steps: int = 50
    save_steps: int = 10000

    keep_latest_n_ckp: int = 8  # 训练过程中，最多保留多少个分数最好的模型文件

    seed: int = 23333
    dataloader_buffer_size: int = 50000
    max_seq_len: int = 256  # 最大句子长度，默认：256


# ======================================================================================
# 以下为模型的配置
@dataclass
class T5ModelConfig:
    d_ff: int = 3072  # 全连接层维度

    d_model: int = 768  # 词向量维度
    num_heads: int = 12  # 注意力头数 d_model // num_heads == d_kv
    d_kv: int = 64  # d_model // num_heads

    num_decoder_layers: int = 10  # Transformer decoder 隐藏层层数
    num_layers: int = 10  # Transformer encoder 隐藏层层数


__all__ = [
    "PROJECT_ROOT",
    "InferConfig",
    "DpoConfig",
    "SFTConfig",
    "TrainConfig",
    "T5ModelConfig",
]
