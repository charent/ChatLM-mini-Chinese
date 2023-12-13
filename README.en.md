# Chat-LM-small

# 1. Introduction
*阅读中文文档 [中文](README.md).*

The parameters of today's large language models tend to be large, and consumer-level computers are relatively slow for simple inference, let alone training a model from scratch.
The goal of this project is to sort out the entire training process of a generative language model, including data cleaning, tokenizer training, model pre-training, SFT instruction fine-tuning, DPO preference optimization, etc. The model parameters of Chat-LM-small are only 0.7B. It can be trained on a machine with a minimum of 16GB of GPU memory (`fp16` or `bf16`). Inference only requires at least 1GB of GPU memory (`bf16`, if you use `int8`, `int4` Quantization, you can also continue to compress).

- Make public all pre-training, SFT instruction fine-tuning, and DPO preference optimization datasets.
- Use the `Huggingface` NLP framework, including `transformers`, `accelerate`, `trl`, `peft`, etc.
- Self-implemented `trainer`, supporting pre-training and SFT fine-tuning on a single machine with a single card or with multiple cards on a single machine. It supports stopping at any position during training and continuing training at any position.
- Pre-training: Integrated into end-to-end `text-to-text` pre-training, non-`mask` mask prediction pre-training.
     - Open source all data cleaning, dataset construction, dataset loading optimization and other processes;
     - tokenizer multi-process word frequency statistics, supports tokenizer training of `sentencepiece` and `huggingface tokenizers`;
     - Pre-training supports breakpoints, and training can be continued from the breakpoint;
     - Streaming loading of large datasets (GB level), supporting buffer data shuffling, does not use memory or hard disk as cache, effectively reducing memory and disk usage. configuring `batch_size=16, max_len=320`, supporting pre-training on a machine with at least 16GB memory + 16GB GPU memory;
     - Training log record.
- SFT fine-tuning: open source SFT dataset and data processing process.
     - The self-implemented `trainer` supports prompt command fine-tuning and supports any breakpoint to continue training;
     - Support `sequence to sequence` fine-tuning of `Huggingface trainer`;
     - Supports traditional low learning rate and only trains fine-tuning of the decoder layer.
- Preference optimization: Use DPO to optimize all preferences.
     - Support using `peft lora` for preference optimization;
     - Supports model merging, `Lora adapter` can be merged into the original model.

<details>
<summary> Recent updates </summary>
   - [2023-12-14] Update Sft and dpo model weight files, update pre-training, SFT and DPO scripts. Update `tokenizer` to `PreTrainedTokenizerFast`, refactor the `dataset` code to support dynamic maximum length. The maximum length of each batch is determined by the longest text of the batch, saving GPU memory. <br/>
   - [2023-12-04] Update the model effect display and update the readme document. <br/>
   - [2023-11-26] Updated dpo training process. <br/>
   - [2023-11-18] The project is open source. <br/>
</details>

# 2. Chat-LM-small model training process
## 2.1 Pre-training dataset
All datasets come from the **Single Round Conversation** dataset published on the Internet. After data cleaning and formatting, they are saved as parquet files. For the data processing process, see `utils/raw_data_process.py`. Main datasets include:

1. Community Q&A json version webtext2019zh-large-scale high-quality dataset, see: [nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus). A total of 4.1 million, with 2.6 million remaining after cleaning.
2. baike_qa2019 encyclopedia Q&A, see: <https://aistudio.baidu.com/datasetdetail/107726>, a total of 1.4 million, and the remaining 1.3 million after waking up.
3. Chinese medical field question and answer dataset, see: [Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data), with a total of 790,000, and the remaining 790,000 after cleaning.
4. ~~Financial industry question and answer data, see: <https://zhuanlan.zhihu.com/p/609821974>, a total of 770,000, and the remaining 520,000 after cleaning. ~~**The data quality is too poor and not used. **
5. Zhihu question and answer data, see: [Zhihu-KOL](https://huggingface.co/datasets/wangrui6/Zhihu-KOL), with a total of 1 million rows, and 970,000 rows remain after cleaning.
6. belle open source instruction training data, introduction: [BELLE](https://github.com/LianjiaTech/BELLE), download: [BelleGroup](https://huggingface.co/BelleGroup), only select `Belle_open_source_1M` , `train_2M_CN`, and `train_3.5M_CN` contain some data with short answers, no complex table structure, and translation tasks (no English vocabulary list), totaling 3.7 million rows, and 3.38 million rows remain after cleaning.
7. Wikipedia entry data, piece together the entries into prompts, the first `N` words of the encyclopedia are the answers, use the encyclopedia data of `202309`, and after cleaning, the remaining 1.19 million entry prompts and answers . Wiki download: [zhwiki](https://dumps.wikimedia.org/zhwiki/), convert the downloaded bz2 file to wiki.txt reference: [WikiExtractor](https://github.com/apertium/WikiExtractor).

The total number of datasets is 10.23 million: Text to Text pre-training set: 9.3 million, evaluation set: 25,000 (because the decoding is slow, the evaluation set is not set too large). ~~Test set: 900,000~~
SFT fine-tuning and DPO optimization datasets are shown below.

## 2.2 Model
T5 model (Text-To-Text Transfer Transformer), for details, see the paper: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683).

The model source code comes from huggingface, see: [T5ForConditionalGeneration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1557).

For model configuration, see `T5ModelConfig` under `config.py`. The official `T5-base`: `encoder layer` and `decoder layer` are both 12 layers. In this project, these two parameters are modified to 10 layers.

Model parameters: 0.7B. Word list size: 29298, including only Chinese and a small amount of English.

## 2.3 Training process
hardware:
```bash
# Pre-training phase:
CPU: 28 vCPU Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
Memory: 60 GB
GPU: RTX A5000 (24GB) * 2

# sft and dpo stages:
CPU: Intel(R) i5-13600k @ 5.1GHz
Memory: 32 GB
GPU: NVIDIA GeForce RTX 4060 Ti 16GB * 1
```
1. **text to text pre-training**: The learning rate is a dynamic learning rate from `1e-4` to `5e-3`, and the pre-training time is 8 days. Training loss:
![traing loss](img/train_loss.png)

2. **prompt supervised fine-tuning (SFT)**: Use the `belle` instruction training dataset (both instruction and answer lengths are below 512), with a dynamic learning rate from `1e-7` to `5e-5` , the fine-tuning time is 2 days. Fine-tuning loss:
![finetune loss](img/sft_loss.png)

3. **dpo direct preference optimization**: dataset [alpaca-gpt4-data-zh](https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh) as `chosen` text , in step `2`, the SFT model performs batch `generate` on the prompts in the dataset, and obtains the `rejected` text, which takes 1 day, dpo full preference optimization, learning rate `le-5`, half precision `fp16`, total `2` `epoch`, taking 3h. dpo loss:
![dpo loss](img/dpo_loss.png)

## 2.4 Dialogue effect display
### 2.4.1 stream chat
By default, `TextIteratorStreamer` of `huggingface transformers` is used to implement streaming dialogue, and only `greedy search` is supported. If you need `beam sample` and other generation methods, please change the `stream_chat` parameter of `cli_demo.py` to `False` .
![](./img/stream_chat.gif)

### 2.4.3 Dialogue display
![](./img/show1.png)

There are problems: the pre-training dataset only has more than 9 million, and the model parameters are only 0.7B. It cannot cover all aspects, and there will be situations where the answer is wrong and the generator is nonsense.

# 三、使用说明
克隆项目：
```bash
git clone --depth 1 https://github.com/charent/Chat-LM-small.git

cd Chat-LM-small
```

## 3.1 安装依赖 
本项目推荐使用`python 3.10`，过老的python版本可能不兼容所依赖的第三方库。

pip安装：
```bash
pip install -r ./requirements.txt
``` 

如果pip安装了CPU版本的pytorch，可以通过下面的命令安装CUDA版本的pytorch：
```bash
# pip 安装torch + cu118
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

conda安装：
```bash
conda install --yes --file ./requirements.txt
```

## 3.2 下载预训练模型及词表
从`Hugging Face Hub`下载模型及文件，需要先安装[Git LFS](https://docs.github.com/zh/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)，然后运行:
```bash 
git clone https://huggingface.co/charent/Chat-LM-small
```
也可以直接从`Hugging Face Hub`仓库[Chat-LM-small](https://huggingface.co/charent/Chat-LM-small)手工下载，将下载的文件移动到`model_save`目录下即可。
    
## 3.3 Text to Text 预训练 
1. 预训练数据集示例
```json
{
    "prompt": "对于花园街，你有什么了解或看法吗？",
    "response": "花园街（是香港油尖旺区的一条富有特色的街道，位于九龙旺角东部，北至界限街，南至登打士街，与通菜街及洗衣街等街道平行。现时这条街道是香港著名的购物区之一。位于亚皆老街以南的一段花园街，也就是\"波鞋街\"整条街约150米长，有50多间售卖运动鞋和运动用品的店舖。旺角道至太子道西一段则为排档区，售卖成衣、蔬菜和水果等。花园街一共分成三段。明清时代，花园街是芒角村栽种花卉的地方。此外，根据历史专家郑宝鸿的考证：花园街曾是1910年代东方殷琴拿烟厂的花园。纵火案。自2005年起，花园街一带最少发生5宗纵火案，当中4宗涉及排档起火。2010年。2010年12月6日，花园街222号一个卖鞋的排档于凌晨5时许首先起火，浓烟涌往旁边住宅大厦，消防接报4"
}
```

2. jupyter-lab or jupyter notebook:

     See the file `train.ipynb`. It is recommended to use jupyter-lab to avoid considering the situation where the terminal process is killed after disconnecting from the server.

3. Console:

    Console training needs to consider that the process will be killed after the connection is disconnected. It is recommended to use the process daemon tool `Supervisor` or `screen` to establish a connection session.

     First, configure `accelerate`, execute the following command, and select according to the prompts. Refer to `accelerate.yaml`, *Note: DeepSpeed installation in Windows is more troublesome*.
     ```bash
     accelerate config
     ```

     Start training. If you want to use the configuration provided by the project, please add the parameter `--config_file ./accelerate.yaml` after the following command `accelerate launch`. *This configuration is based on the single-machine 2xGPU configuration. *

     *There are two scripts for pre-training. The trainer implemented in this project corresponds to `train.py`, and the trainer implemented by huggingface corresponds to `pre_train.py`. You can use either one and the effect will be the same. The training information display of the trainer implemented in this project is more beautiful, and it is easier to modify the training details (such as loss function, log records, etc.). All support breakpoints to continue training. The trainer implemented in this project supports continuing training after a breakpoint at any position. Press ` ctrl+c` will save the breakpoint information when exiting the script. *

     Single machine and single card:
     ```bash
     # The trainer implemented in this project
     accelerate launch ./train.py train

     # Or use huggingface trainer
     python pre_train.py
     ```

     Single machine with multiple cards:
     ```bash
     # The trainer implemented in this project
     accelerate launch --multi_gpu --num_processes 2 ./train.py train

     # Or use huggingface trainer
     python pre_train.py
     ```

     Continue training from the breakpoint:
     ```
     # The trainer implemented in this project
     accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_keep_training=True

     # Or use huggingface trainer
     # You need to add `resume_from_checkpoint=True` to the `train` function in `pre_train.py`
     python pre_train.py
     ```

## 3.4 SFT fine-tuning
The SFT dataset all comes from the contribution of [BELLE](https://github.com/LianjiaTech/BELLE). Thank you. The SFT datasets are: [generated_chat_0.4M](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M), [train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN ) and [train_2M_CN](https://huggingface.co/datasets/BelleGroup/train_2M_CN), about 1.37 million rows remain after cleaning.
Example of fine-tuning dataset with sft command:
```json
{
    "prompt": "解释什么是欧洲启示录",
    "response": "欧洲启示录（The Book of Revelation）是新约圣经的最后一卷书，也被称为《启示录》、《默示录》或《约翰默示录》。这本书从宗教的角度描述了世界末日的来临，以及上帝对世界的审判和拯救。 书中的主题包括来临的基督的荣耀，上帝对人性的惩罚和拯救，以及魔鬼和邪恶力量的存在。欧洲启示录是一个充满象征和暗示的文本，对于解读和理解有许多不同的方法和观点。"
}
```
Make your own dataset by referring to the sample `parquet` file in the `data` directory. The dataset format is: the `parquet` file is divided into two columns, one column of `prompt` text, representing the prompt, and one column of `response` text, representing the expected model. output.
For fine-tuning details, see the `train` method under `model/trainer.py`. When `is_finetune` is set to `True`, fine-tuning will be performed. Fine-tuning will freeze the embedding layer and encoder layer by default, and only train the decoder layer. If you need to freeze other parameters, please adjust the code yourself.

Run SFT fine-tuning:
```bash
# For the trainer implemented in this project, just add the parameter `--is_finetune=True`. The parameter `--is_keep_training=True` can continue training from any breakpoint.
accelerate launch --multi_gpu --num_processes 2 ./train.py --is_finetune=True

# Or use huggingface trainer
python sft_train.py
```

## 3.5 RLHF (Reinforcement Learning Human Feedback Optimization Method)
### 3.5.1 Preference optimization method

Here are two common preferred methods: PPO and DPO. Please search papers and blogs for specific implementations.

1. PPO method (approximate preference optimization, Proximal Policy Optimization)
     Step 1: Use the fine-tuning dataset to do supervised fine-tuning (SFT, Supervised Finetuning).
     Step 2: Use the preference dataset (a prompt contains at least 2 responses, one wanted response and one unwanted response. Multiple responses can be sorted by score, with the most wanted one having the highest score) to train the reward model (RM, Reward Model). You can use the `peft` library to quickly build the Lora reward model.
     Step 3: Use RM to perform supervised PPO training on the SFT model so that the model meets preferences.

2. Use DPO (Direct Preference Optimization) fine-tuning (**This project uses the DPO fine-tuning method, which saves GPU memory**)
     On the basis of obtaining the SFT model, there is no need to train the reward model, and fine-tuning can be started by obtaining the positive answer (chosen) and the negative answer (rejected). The fine-tuned `chosen` text comes from the original dataset [alpaca-gpt4-data-zh](https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh), and the rejected text `rejected` comes from SFT Model output after fine-tuning 1 epoch, two other datasets: [huozi_rlhf_data_json](https://huggingface.co/datasets/Skepsun/huozi_rlhf_data_json) and [rlhf-reward-single-round-trans_chinese](https:// huggingface.co/datasets/beyond/rlhf-reward-single-round-trans_chinese), a total of 80,000 dpo data after the merger.
    
     For the dpo dataset processing process, see `utils/dpo_data_process.py`.
    
DPO preference optimization dataset example:
```json
    {
        "prompt": "为给定的产品创建一个创意标语。，输入：可重复使用的水瓶。",
        "chosen": "\"保护地球，从拥有可重复使用的水瓶开始！\"",
        "rejected": "\"让你的水瓶成为你的生活伴侣，使用可重复使用的水瓶，让你的水瓶成为你的伙伴\""
    },
```
Run preference optimization:
```bash
pythondpo_train.py
```

## 3.6 Reasoning
Make sure there are the following files in the `model_save` directory:
```bash
Chat-LM-small
├─model_save
│ ├─chat_lm_t5.pre7.sft9w.dpo6k.bin
| └─tokenizer
| ├─special_tokens_map.json
| ├─tokenizer.json
| └─tokenizer_config.json
```
Only the `chat_lm_t5.pre7.sft9w.dpo6k.bin` file needs to be downloaded manually. The files in the `tokenizer` folder are included when cloning the Git repository.

1. Console run:
```bash
python cli_demo.py
```

2. API call
```
python api_demo.py
```

API call example:
API调用示例：
```bash
curl --location '127.0.0.1:8812/api/chat' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer Bearer' \
--data '{
    "input_txt": "感冒了要怎么办"
}'
```
![api demo](./img/api_example.png)

# 4. Quote
If you think this project is helpful to you, please quote it.
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

# 5. Other matters
This project does not bear any risks and responsibilities arising from data security and public opinion risks caused by open source models and codes, or any model being misled, abused, disseminated, or improperly exploited.
