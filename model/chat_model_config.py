from transformers import T5Config

class TextToTextModelConfig(T5Config):
    model_type = 't5'