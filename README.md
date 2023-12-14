# Chat-LM-small

# 一、介绍
*Read this in [English](README.en.md).*
现在的大语言模型的参数往往较大，消费级电脑单纯做推理都比较慢，更别说想自己从头开始训练一个模型了。 本项目的目标是梳理整套生成式语言模型的训练流程，包括数据清洗、tokenizer训练、模型预训练、SFT指令微调、RLHF优化等。 

Chat-LM-small为中文对话小模型，模型参数只有210M（0.2B），可以在最低4GB显存的机器进行预训练（`batch_size=1`，`fp16`或者` bf16`），`float16`加载、推理最少只需要512MB显存。 


- 公开所有预训练、SFT指令微调、DPO偏好优化数据集。
- 使用`Huggingface`NLP框架，包括`transformers`、`accelerate`、`trl`、`peft`等。
- 自实现`trainer`，支持单机单卡、单机多卡进行预训练、SFT微调。训练过程中支持在任意位置停止，及在任意位置继续训练。
- 预训练：整合为端到端的`text-to-text`预训练，非`mask`掩码预测预训练。
    - 开源所有数据清洗、数据集构造、数据集加载优化等流程；
    - tokenizer多进程词频统计，支持`sentencepiece`、`huggingface tokenizers`的tokenizer训练；
    - 预训练支持任意位置断点，可从断点处继续训练;
    - 大数据集（GB级别）流式加载、支持缓冲区数据打乱，不利用内存、硬盘作为缓存，有效减少内存、磁盘占用。配置`batch_size=1, max_len=320`下，最低支持在16GB内存+4GB显存的机器上进行预训练；
    - 训练日志记录。
- SFT微调：开源SFT数据集及数据处理过程。
    - 自实现`trainer`支持prompt指令微调， 支持任意断点继续训练；
    - 支持`Huggingface trainer`的`sequence to sequence`微调；
    - 支持传统的低学习率，只训练decoder层的微调。
- 偏好优化：使用DPO进行全量偏好优化。
    - 支持使用`peft lora`进行偏好优化；
    - 支持模型合并，可将`Lora adapter`合并到原始模型中。

🟢**最近更新**
<details close> 
<summary>  <b>2023-12-14</b> </summary>
- 更新SFT、DPO后的模型权重文件。 <br/>
- 更新预训练、SFT及DPO脚本。 <br/>
- 更新`tokenizer`为`PreTrainedTokenizerFast`。 <br/>
- 重构`dataset`代码，支持动态最大长度，每个批次的最大长度由该批次的最长文本决定，节省显存。 <br/>
</details>

<details close> 
<summary> <b>2023-12-04</b> </summary>
- 更新`generate`参数及模型效果展示。<br/>
- 更新readme文档。<br/>
</details>

<details close> 
<summary> <b>2023-11-28</b> </summary>
- 更新dpo训练代码及模型权重。<br/>
</details>

<details close> 
<summary> <b>2023-10-19</b> </summary>
- 项目开源， 开放模型权重供下载。 <br/>
</details>


# 二、Chat-LM-small模型训练过程
## 2.1 预训练数据集
所有数据集均来自互联网公开的**单轮对话**数据集，经过数据清洗、格式化后保存为parquet文件。数据处理过程见`utils/raw_data_process.py`。主要数据集包括： 

1. 社区问答json版webtext2019zh-大规模高质量数据集，见：[nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus)。共410万，清洗后剩余260万。
2. baike_qa2019百科类问答，见：<https://aistudio.baidu.com/datasetdetail/107726>，共140万，清醒后剩余130万。
3. 中国医药领域问答数据集，见：[Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data)，共79万，清洗后剩余79万。
4. ~~金融行业问答数据，见：<https://zhuanlan.zhihu.com/p/609821974>，共77万，清洗后剩余52万。~~**数据质量太差，未采用。**
5. 知乎问答数据，见：[Zhihu-KOL](https://huggingface.co/datasets/wangrui6/Zhihu-KOL)，共100万行，清洗后剩余97万行。
6. belle开源的指令训练数据，介绍：[BELLE](https://github.com/LianjiaTech/BELLE)，下载：[BelleGroup](https://huggingface.co/BelleGroup)，仅选取`Belle_open_source_1M`、`train_2M_CN`、及`train_3.5M_CN`中部分回答较短、不含复杂表格结构、翻译任务（没做英文词表）的数据，共370万行，清洗后剩余338万行。
7. 维基百科（Wikipedia）词条数据，将词条拼凑为提示语，百科的前`N`个词为回答，使用`202309`的百科数据，清洗后剩余119万的词条提示语和回答。Wiki下载：[zhwiki](https://dumps.wikimedia.org/zhwiki/)，将下载的bz2文件转换为wiki.txt参考：[WikiExtractor](https://github.com/apertium/WikiExtractor)。 

数据集总数量1023万：Text to Text预训练集：930万，评估集：2.5万（因为解码较慢，所以没有把评估集设置太大）。~~测试集：90万。~~ 
SFT微调和DPO优化数据集见下文。

## 2.2 模型
T5模型（Text-To-Text Transfer Transformer），详情见论文: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)。

模型源码来自huggingface，见：[T5ForConditionalGeneration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1557)。

模型配置见[model_config.json](https://huggingface.co/charent/Chat-LM-small/blob/main/model_config.json)，官方的`T5-base`：`encoder layer`和`decoder layer `均为为12层，本项目这两个参数修改为10层。 

模型参数：210M。词表大小：29298，仅包含中文和少量英文。

## 2.3 训练过程
硬件：
```bash
# 预训练阶段：
CPU: 28 vCPU Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
内存：60 GB
显卡：RTX A5000(24GB) * 2

# sft及dpo阶段：
CPU: Intel(R) i5-13600k @ 5.1GHz
内存：32 GB
显卡：NVIDIA GeForce RTX 4060 Ti 16GB * 1
```
1. **text to text 预训练**：学习率为`1e-4`到`5e-3`的动态学习率，预训练时间为8天。训练损失： 

![traing loss](img/train_loss.png) 

1. **prompt监督微调（SFT）**：使用`belle`指令训练数据集（指令和回答长度都在512以下），学习率为`1e-7`到`5e-5`的动态学习率，微调时间2天。微调损失： 
   
![finetune loss](img/sft_loss.png) 

1. **dpo直接偏好优化**：数据集[alpaca-gpt4-data-zh](https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh)作为`chosen`文本，步骤`2`中SFT模型对数据集中的prompt做批量`generate`，得到`rejected`文本，耗时1天，dpo全量偏好优化，学习率`le-5`，半精度`fp16`,共`2`个`epoch`，耗时3h。dpo损失： 
 
![dpo loss](img/dpo_loss.png) 

## 2.4 对话效果展示
### 2.4.1 stream chat
默认使用`huggingface transformers`的 `TextIteratorStreamer`实现流式对话，只支持`greedy search`，如果需要`beam sample`等其他生成方式，请将`cli_demo.py`的`stream_chat`参数修改为`False`。
![](./img/stream_chat.gif)

### 2.4.3 对话展示
![](./img/show1.png)

存在问题：预训练数据集只有900多万，模型参数也仅210M，不能涵盖所有方面，会有答非所问、废话生成器的情况。

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

## 3.2 下载预训练模型及模型配置文件

从`Hugging Face Hub`下载模型权重及配置文件，需要先安装[Git LFS](https://docs.github.com/zh/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)，然后运行: 

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
   
2. jupyter-lab 或者 jupyter notebook:  

    见文件`train.ipynb`，推荐使用jupyter-lab，避免考虑与服务器断开后终端进程被杀的情况。 

3. 控制台： 

   控制台训练需要考虑连接断开后进程被杀的，推荐使用进程守护工具`Supervisor`或者`screen`建立连接会话。

    首先要配置`accelerate`，执行以下命令， 根据提示选择即可，参考`accelerate.yaml`，*注意：DeepSpeed在Windows安装比较麻烦*。
    ``` bash
    accelerate config
    ```

    开始训练，如果要使用工程提供的配置请在下面的命令`accelerate launch`后加上参数`--config_file ./accelerate.yaml`，*该配置按照单机2xGPU配置。* 

    *预训练有两个脚本，本项目实现的trainer对应`train.py`，huggingface实现的trainer对应`pre_train.py`，用哪个都可以，效果一致。本项目实现的trainer训练信息展示更美观、更容易修改训练细节（如损失函数，日志记录等），均支持断点继续训练，本项目实现的trainer支持在任意位置断点后继续训练，按`ctrl+c`退出脚本时会保存断点信息。* 

    单机单卡：
    ``` bash
    # 本项目实现的trainer
    accelerate launch ./train.py train

    # 或者使用 huggingface trainer
    python pre_train.py
    ```

    单机多卡：
    ``` bash
    # 本项目实现的trainer
    accelerate launch --multi_gpu --num_processes 2 ./train.py train

    # 或者使用 huggingface trainer
    python pre_train.py
    ```

    从断点处继续训练：
    ```
    # 本项目实现的trainer
    accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_keep_training=True

    # 或者使用 huggingface trainer
    # 需要在`pre_train.py`中的`train`函数添加`resume_from_checkpoint=True`
    python pre_train.py
    ```

## 3.4 SFT微调 
SFT数据集全部来自[BELLE](https://github.com/LianjiaTech/BELLE)大佬的贡献，感谢。SFT数据集分别为：[generated_chat_0.4M](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)、[train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)和[train_2M_CN](https://huggingface.co/datasets/BelleGroup/train_2M_CN)，清洗后剩余约137万行。
sft指令微调数据集示例：
```json
{
    "prompt": "解释什么是欧洲启示录",
    "response": "欧洲启示录（The Book of Revelation）是新约圣经的最后一卷书，也被称为《启示录》、《默示录》或《约翰默示录》。这本书从宗教的角度描述了世界末日的来临，以及上帝对世界的审判和拯救。 书中的主题包括来临的基督的荣耀，上帝对人性的惩罚和拯救，以及魔鬼和邪恶力量的存在。欧洲启示录是一个充满象征和暗示的文本，对于解读和理解有许多不同的方法和观点。"
}
```

参考`data`目录下的示例`parquet`文件制作自己的数据集，数据集格式：`parquet`文件分两列，一列`prompt`文本，表示提示语，一列`response`文本，表示期待的模型输出。
微调细节见`model/trainer.py`下的`train`方法, `is_finetune`设置为`True`时，将进行微调，微调默认会冻结embedding层和encoder层，只训练decoder层。如需要冻结其他参数，请自行调整代码。 

运行SFT微调：
``` bash
# 本项目实现的trainer， 添加参数`--is_finetune=True`即可, 参数`--is_keep_training=True`可从任意断点处继续训练
accelerate launch --multi_gpu --num_processes 2 ./train.py --is_finetune=True

# 或者使用 huggingface trainer
python sft_train.py
```

## 3.5 RLHF（强化学习人类反馈优化方法）
### 3.5.1 偏好优化方法

偏好方法这里介绍常见的两种：PPO和DPO，具体实现请自行搜索论文及博客。

1.  PPO方法（近似偏好优化,Proximal Policy Optimization）
    步骤1：使用微调数据集做有监督微调（SFT， Supervised Finetuning）。 
    步骤2：使用偏好数据集（一个prompt至少包含2个回复，一个想要的回复，一个不想要的回复。多个回复可以按照分数排序，最想要的分数最高）训练奖励模型（RM， Reward Model）。可使用`peft`库快速搭建Lora奖励模型。 
    步骤3：利用RM对SFT模型进行有监督PPO训练，使得模型满足偏好。 

2.  使用DPO（直接偏好优化，Direct Preference Optimization）微调（**本项目采用DPO微调方法，比较节省显存**）
    在获得SFT模型的基础上，无需训练奖励模型，取得正向回答（chosen）和负向回答（rejected）即可开始微调。微调的`chosen`文本来自原数据集[alpaca-gpt4-data-zh](https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh)，拒绝文本`rejected`来自SFT微调1个epoch后的模型输出，另外两个数据集：[huozi_rlhf_data_json](https://huggingface.co/datasets/Skepsun/huozi_rlhf_data_json)和[rlhf-reward-single-round-trans_chinese](https://huggingface.co/datasets/beyond/rlhf-reward-single-round-trans_chinese)，合并后共8万条dpo数据。
    
    dpo数据集处理过程见`utils/dpo_data_process.py`。
    
DPO偏好优化数据集示例：
```json
    {
        "prompt": "为给定的产品创建一个创意标语。，输入：可重复使用的水瓶。",
        "chosen": "\"保护地球，从拥有可重复使用的水瓶开始！\"",
        "rejected": "\"让你的水瓶成为你的生活伴侣，使用可重复使用的水瓶，让你的水瓶成为你的伙伴\""
    },
```

运行偏好优化：
``` bash
python dpo_train.py
```

## 3.6 推理 
确保`model_save`目录下有以下文件：
```bash
Chat-LM-small
├─model_save
│  ├─chat_lm_t5.pre7.sft9w.dpo6k.bin
|  ├─model_config.json
|  └─tokenizer
|     ├─special_tokens_map.json
|     ├─tokenizer.json
|     └─tokenizer_config.json
```
文件`chat_lm_t5.pre7.sft9w.dpo6k.bin`和`model_config.json`需要手工下载，`tokenizer`文件夹下的文件在克隆Git仓库的时候自带。

1. 控制台运行：
```bash
python cli_demo.py
```

2. API调用
```
python api_demo.py
```

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


# 四、引用
如果你觉得本项目对你有所帮助，欢迎引用。
```conf
@misc{Charent2023,
    author={Charent Chen},
    title={A small chinese chatbot with 210M parameters base on T5 model},
    year={2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/charent/Chat-LM-small}},
}
```

# 五、其他事项
本项目不承担开源模型和代码导致的数据安全、舆情风险或发生任何模型被误导、滥用、传播、不当利用而产生的风险和责任。

<!-- # 提示
```bash
# 导出项目依赖的包：
pipreqs --encoding "utf-8" --force
``` -->

