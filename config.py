from dataclasses import dataclass
from os.path import dirname, abspath

PROJECT_ROOT: str = abspath(dirname(__file__))

@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 4
    learn_rate: float = 0.001

    mixed_precision: str = "no" #混合精度 ''no','fp16','bf16 or 'fp8'

    tokenizer_file: str = PROJECT_ROOT + '/model_save/my_merged_tokenizer.json'
    model_file: str= PROJECT_ROOT + '/model_save/chat_small_t5.pth'
    train_file: str = PROJECT_ROOT + '/data/my_test_dataset.parquet'
    validation_file: str = PROJECT_ROOT + '/data/my_valid_dataset.parquet'
    test_file: str = PROJECT_ROOT + '/data/my_valid_dataset.parquet'
    trainer_log_file: str = PROJECT_ROOT + '/logs/trainer.log'

    seed: int = 23333
    dataloader_buffer_size: int = 8192
    max_seq_len: int = 128


#==================================================================


@dataclass
class T5ModelConfig:
    max_seq_len: int = 128
    d_ff: int = 128                     # 全连接层维度
    d_kv: int = 16                      # 注意力头数 d_model // num_heads:  int == d_kv
    d_model: int = 128                  # 词向量大学
    num_decoder_layers:  int = 2        # Transformer decoder 隐藏层层数
    num_heads: int = 8
    num_layers: int = 2                 #  Transformer encoder 隐藏层层数