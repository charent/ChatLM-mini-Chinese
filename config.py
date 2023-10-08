from dataclasses import dataclass
from os.path import dirname, abspath

PROJECT_ROOT: str = abspath(dirname(__file__))

@dataclass
class TrainConfig:
    epochs: int = 6
    batch_size_per_gpu: int = 20
    
    learn_rate: float = 0.0001    # 最大 div_factor * learn_rate
    div_factor: int = 50

    mixed_precision: str = "bf16" #混合精度 ''no','fp16','bf16' or 'fp8'

    tokenizer_file: str = PROJECT_ROOT + '/model_save/my_merged_tokenizer.json'
    model_file: str= PROJECT_ROOT + '/model_save/chat_small_t5_{}.pth'
    model_config_file: str= PROJECT_ROOT + '/model_save/model_config.json'
    train_file: str = PROJECT_ROOT + '/data/my_train_dataset.parquet'
    validation_file: str = PROJECT_ROOT + '/data/my_valid_dataset.parquet'
    test_file: str = PROJECT_ROOT + '/data/my_test_dataset.parquet'

    # dataset_cache_dir: str = PROJECT_ROOT + '/data/.cache'
    # trainer_log_file: str = PROJECT_ROOT + '/logs/trainer.log'

    seed: int = 23333
    dataloader_buffer_size: int = 50000
    max_seq_len: int = 256                      # 最大句子长度，默认：256


#==================================================================


@dataclass
class T5ModelConfig:

    d_ff: int = 3072                        # 全连接层维度，默认：2048, 大：3072

    d_model: int = 768                      # 词向量维度，默认：512, 大：768
    num_heads: int = 12                     # 注意力头数 d_model // num_heads == d_kv， 默认：8, 大：12
    d_kv: int = 64                          # d_model // num_heads， 默认：64, 大：64

    num_decoder_layers: int = 10            # Transformer decoder 隐藏层层数， 默认：6, 大：10
    num_layers: int = 10                    # Transformer encoder 隐藏层层数，默认：6, 大：10