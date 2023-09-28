from dataclasses import dataclass

@dataclass
class TrainConfig:
    lr: float = 0.001


@dataclass
class ModelConfig:
    max_seq_len: int = 128
    d_ff: int = 128                     # 全连接层维度
    d_kv: int = 16                      # 注意力头数 d_model // num_heads:  int == d_kv
    d_model: int = 128                  # 词向量大学
    num_decoder_layers:  int = 2        # Transformer decoder 隐藏层层数
    num_heads: int = 8
    num_layers: int = 2                 #  Transformer encoder 隐藏层层数
    vocab_size: int = 4096              # 词库大大小
    