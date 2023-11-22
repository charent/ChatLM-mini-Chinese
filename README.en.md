# Chat-LM-small

# 一、Introduction
*阅读中文文档 [中文](README.md).*
The parameters of today's large language models tend to be large, and consumer-level computers are relatively slow for simple inference, let alone training a model from scratch. The original intention of this project is to sort out the entire training process of the generative language model, from data cleaning, tokenizer training, model pre-training, model fine-tuning to the final product, and form its own engineering system instead of simply calling `from_pretrained`. The model parameters of this project are only 0.7B. It can be trained on a machine with a minimum of 16G of GPU memory (`fp16` or `bf16`). Inference only requires a minimum of 1.4G of GPU memory (`fp8`. If you do `int4` quantization, you can continue to compress ).

# 二、Chat-LM-small Model training process
## 2.1 Datasets
All datasets come from the **Single Round Conversation** dataset published on the Internet. After data cleaning and formatting, they are saved as parquet files. For the data processing process, see `utils/raw_data_process.py`. Main datasets include:

1. Community Q&A json version webtext2019zh-large-scale high-quality dataset, see: [nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus). A total of 4.1 million, with 2.6 million remaining after cleaning.
2. baike_qa2019 encyclopedia Q&A, see: <https://aistudio.baidu.com/datasetdetail/107726>, a total of 1.4 million, and the remaining 1.3 million after waking up.
3. Chinese medical field question and answer dataset, see: [Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data), with a total of 790,000, and the remaining 790,000 after cleaning.
4. ~~Financial industry question and answer data, see: <https://zhuanlan.zhihu.com/p/609821974>, a total of 770,000, and the remaining 520,000 after cleaning.~~ **The data quality is too poor and not used.**
5. Zhihu question and answer data, see: [Zhihu-KOL](https://huggingface.co/datasets/wangrui6/Zhihu-KOL), with a total of 1 million rows, and 970,000 rows remain after cleaning.
6. belle open source instruction training data, introduction: [BELLE](https://github.com/LianjiaTech/BELLE), download: [BelleGroup](https://huggingface.co/BelleGroup), only select `Belle_open_source_1M`, `train_2M_CN`, and The data in `train_3.5M_CN` contains short answers, does not contain complex table structures, and does not include translation tasks (no English word list). There are a total of 3.7 million rows, and 3.38 million rows remain after cleaning.
7. Wikipedia entry data, the entries are pieced together into prompts, and the first `N` words in the encyclopedia are answers. Using `202309` encyclopedia data, 1.19 million entry prompts and answers remain after cleaning. Wiki download: [zhwiki](https://dumps.wikimedia.org/zhwiki/), convert the downloaded bz2 file to wiki.txt reference: [WikiExtractor](https://github.com/apertium/WikiExtractor).

Data summary: The total number of datasets is 10.23 million: training set: 9.3 million, evaluation set: 25,000 (because the decoding is slow, the evaluation set is not set too large), test set: 900,000.

## 2.2 Model
T5 model (Text-To-Text Transfer Transformer), for details, see the paper: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683).

The model source code comes from huggingface, see: [T5ForConditionalGeneration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1557).

For model configuration, see `T5ModelConfig` under `config.py`. The official `T5-base`: `encoder layer` and `decoder layer` are both 12 layers. In this project, these two parameters are modified to 10 layers.

Model parameters: 0.7B. Word list size: 29298, including only Chinese and a small amount of English.

## 2.3 training process
hardware:
```bash
CPU: 28 vCPU Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
Memory: 60 GB
GPU: RTX A5000 (24GB) * 2
```
1. Pre-training: The learning rate is a dynamic learning rate from `1e-4` to `5e-3`, and the training time is 8 days. Training loss:
![traing loss](img/train_loss.png)

2. Prompt fine-tuning: Use the `belle` command to train the dataset (the command and answer lengths are both below 320), the learning rate is a dynamic learning rate from `1e-5` to `1e-4`, freeze the `encoder` parameters, and only finetune the `decoder` parameters, the fine-tuning time is 1 day. Fine-tuning loss:
![finetune loss](img/finetune_loss.png)

## 2.4 Dialogue effect display

![](./img/show1.png)
![](./img/show2.png)
![](./img/show3.png)
![](./img/show4.png)

There are problems: the pre-trained dataset is only more than 9 million, which cannot cover all aspects, and there will be cases of incorrect answers and nonsense generators.

# 三、Instructions for using
Clone project:
```bash
git clone --depth 1 https://github.com/charent/Chat-LM-small.git

cd Chat-LM-small
```

## Install dependencies 

It is recommended to use `python 3.10` for this project. Older python versions may not be compatible with the third-party libraries it depends on.

pip install：
```bash
pip install -r ./requirements.txt
``` 

If pip installed the CPU version of pytorch, you can install the CUDA version of pytorch with the following command:
```bash
# pip install torch + cu118
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

conda install：：
```bash
conda install --yes --file ./requirements.txt
```

## Download the pre-trained model and vocabulary

To download models and files from `Hugging Face Hub`, you need to install [Git LFS](https://docs.github.com/zh/repositories/working-with-files/managing-large-files/installing-git-large-file-storage), then run:
```bash
git clone https://huggingface.co/charent/Chat-LM-small
```
You can also manually download it directly from the `Hugging Face Hub` repository [Chat-LM-small](https://huggingface.co/charent/Chat-LM-small) and move the downloaded file to the `model_save` directory. 
    
## Training
   
1. jupyter-lab or jupyter notebook:  

    See the file `train.ipynb`. It is recommended to use jupyter-lab to avoid considering the situation where the terminal process is killed after disconnecting from the server. 

2. console： 

    Console training needs to consider that the process will be killed after the connection is disconnected. It is recommended to use the process daemon tool `Supervisor` or `screen` to establish a connection session.

    First, configure `accelerate`, execute the following command, and select according to the prompts. Refer to `accelerate.yaml`, *Note: DeepSpeed installation in Windows is more troublesome*.

    ``` bash
    accelerate config
    ```
    Start training. If you want to use the configuration provided by the project, please add the parameter `--config_file ./accelerate.yaml` after the following command `accelerate launch`. *This configuration is based on the single-machine 2xGPU configuration*

    Single machine with single GPU:
    ``` bash
    accelerate launch ./train.py train
    ```

    Single machine with multiple GPUs:
    ``` bash
    accelerate launch --multi_gpu --num_processes 2 ./train.py train
    ```

    Continue training from the breakpoint:
    ```
    accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_keep_training=True
    ```

## Fine-tuning

#### TO DO
> Use RLHF (reinforcement learning and human feedback method) for fine-tuning
> Step 1: Use the fine-tuning dataset to do supervised fine-tuning (SFT, Supervised Finetuning).
> Step 2: Use the preference dataset (a prompt contains at least 2 responses, one wanted response and one unwanted response. Multiple responses can be sorted by score, the most wanted one has the highest score) to train the reward model (RM, Reward Model).You can use the `peft` library to quickly build a Lora reward model.
> Step 3: Use RM to perform supervised PPO training on the SFT model (DPO training can be used if there is insufficient GPU memory) to make the model meet preferences.
   
Make your own dataset by referring to the sample `parquet` file in the `data` directory. The dataset format is: the `parquet` file is divided into two columns, one column of `prompt` text, representing the prompt, and another column of `response` text, representing the expected model. output.

For fine-tuning details, see the `train` method under `model/trainer.py`. When `is_finetune` is set to `True`, fine-tuning will be performed. Fine-tuning will freeze the embedding layer and encoder layer by default, and only train the decoder layer. If you need to freeze other parameters, please adjust the code yourself.
**Fine-tuning Notes**: The learning rate should be lower than `1e-5`, and it is recommended to use `fp16` for mixed precision, otherwise `loss` may be `Nan`

    ``` bash
    accelerate launch --multi_gpu --num_processes 2 ./train.py --is_finetune=True
    ```
## Inference 
Make sure there are the following files in the `model_save` directory:
```bash
chat_small_t5.best.bin
model_config.json
my_merged_tokenizer.json
```
1. running on console:
```bash
python cli_demo.py
```

1. running with API
```
python api_demo.py
```
example of calling chat api：
```bash
curl --location '127.0.0.1:8812/api/chat' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer Bearer' \
--data '{
    "input_txt": "感冒了要怎么办？"
}'
```
![api demo](./img/api_example.png)


# Cite
```conf
@misc{Charent2023,
    author={Charent Chen},
    title={A small chinese chatbot with 0.7B parameters base on T5 model},
    year={2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/charent/Chat-LM-small}},
}
```

# Notes：
This project does not bear any risks and responsibilities arising from data security and public opinion risks caused by open source models and codes, or any model being misled, abused, disseminated, or improperly exploited.


