from dataclasses import dataclass
from os.path import dirname, abspath

# replace '\' on windows to '/'
PROJECT_ROOT: str = '/'.join(abspath(dirname(__file__)).split('\\')) if '\\' in abspath(dirname(__file__)) else abspath(dirname(__file__))

# ===================================================================================
# 以下为推断的配置
@dataclass
class InferConfig:
    max_seq_len: int = 320                          # 回答的最大长度
    mixed_precision: str = "bf16"                   # 混合精度 ''no','fp16','bf16' or 'fp8'

    # 全量DPO模型文件
    model_file: str = PROJECT_ROOT + '/model_save/chat_small_t5.best.dpo.bin'

    # lora PDO 合并后的模型文件
    # model_file: str = PROJECT_ROOT + '/model_save/chat_small_t5.best.dpo.lora_merged.bin'

    model_config_file: str = PROJECT_ROOT + '/model_save/model_config.json'
    tokenizer_file: str = PROJECT_ROOT + '/model_save/my_merged_tokenizer.json'
    
    #======================================
    # this confing for api demo:
    api_key: str = ""
    host: str = '127.0.0.1'
    port: int = 8812
    reload: bool = True
    workers: int = 1
    log_level: str = 'info'
    #======================================


#===================================================================================
# 以下为dpo训练配置
@dataclass
class DpoConfig:
    max_seq_len: int = 320  
    mixed_precision: str = "fp8"
    sft_model_file: str = PROJECT_ROOT + '/model_save/chat_small_t5.best.bin'
    model_config_file: str = PROJECT_ROOT + '/model_save/model_config.json'
    tokenizer_file: str = PROJECT_ROOT + '/model_save/my_merged_tokenizer.json'
    dpo_train_file: str = PROJECT_ROOT + '/data/dpo_train.json'
    dpo_eval_file: str = PROJECT_ROOT + '/data/dpo_eval.json'
    adapter_file: str = PROJECT_ROOT + '/data/adapter_model.safetensors'

    per_device_train_batch_size: int = 10
    max_steps: int = 2048
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    evaluation_strategy: str = "steps"
    logging_first_step: bool = True
    logging_steps: int = 10                      
    eval_steps: int = 500
    output_dir: str = PROJECT_ROOT + '/model_save/dpo'
    warmup_steps: int = 50
    fp16: bool = True
    seed: int = 23333
    beta: float = 0.1

# ===================================================================================
# 以下为训练的配置
@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size_per_gpu: int = 32
    
    learn_rate: float = 0.0001                      # 最大 div_factor * learn_rate
    div_factor: int = 50

    mixed_precision: str = "bf16"                   # 混合精度 ''no','fp16','bf16' or 'fp8'

    # 注意：计算梯度时相当于batch_size * gradient_accumulation_steps，说人话就是梯度累积步数>1时，等于增大n倍的batch_size
    gradient_accumulation_steps: int = 8           # 累积梯度更新步数

    warmup_steps: int = 1024                        # 模型参数预热步数，预热样本数=warmup_steps * batch_size * gradient_accumulation_steps

    tokenizer_file: str = PROJECT_ROOT + '/model_save/my_merged_tokenizer.json'
    model_file: str = PROJECT_ROOT + '/model_save/chat_small_t5.{}.bin'
    model_config_file: str = PROJECT_ROOT + '/model_save/model_config.json'
    train_file: str = PROJECT_ROOT + '/data/my_train_dataset.parquet'
    validation_file: str = PROJECT_ROOT + '/data/my_valid_dataset.parquet'
    test_file: str = PROJECT_ROOT + '/data/my_test_dataset.parquet'

    # 从哪个模型开始微调，仅当traing 函数 is_finetune = True时生效
    # 微调记得冻结某些层或者调低学习率
    finetune_from_ckp_file = PROJECT_ROOT + '/model_save/chat_small_t5.best.bin'

    # 训练状态保存，中断后可以从此处继续训练
    train_state_dir: str = PROJECT_ROOT + '/model_save/train_latest_state'

    # dataset_cache_dir: str = PROJECT_ROOT + '/data/.cache'
    # trainer_log_file: str = PROJECT_ROOT + '/logs/trainer.log'

    keep_latest_n_ckp: int = 8                  # 训练过程中，最多保留多少个分数最好的模型文件

    seed: int = 23333
    dataloader_buffer_size: int = 50000
    max_seq_len: int = 256                      # 最大句子长度，默认：256


#======================================================================================
# 以下为模型的配置
@dataclass
class T5ModelConfig:

    d_ff: int = 3072                        # 全连接层维度，默认：2048, 大：3072

    d_model: int = 768                      # 词向量维度，默认：512, 大：768
    num_heads: int = 12                     # 注意力头数 d_model // num_heads == d_kv， 默认：8, 大：12
    d_kv: int = 64                          # d_model // num_heads， 默认：64, 大：64

    num_decoder_layers: int = 10            # Transformer decoder 隐藏层层数， 默认：6, 大：10
    num_layers: int = 10                    # Transformer encoder 隐藏层层数，默认：6, 大：10