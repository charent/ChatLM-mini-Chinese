import fire

from config import  TrainConfig, T5ModelConfig
from model.trainer import ChatTrainer


if __name__ == '__main__':
    train_config = TrainConfig()
    model_config = T5ModelConfig()

    chat_trainer = ChatTrainer(train_config=train_config, model_config=model_config)

    # 解析命令行参数，执行指定函数
    # e.g: python train.py train
    fire.Fire(component=chat_trainer)