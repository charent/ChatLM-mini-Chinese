# Chat-LM-small

# 一、介绍
*Read this in [English](README.en.md).*
现在的大语言模型的参数往往较大，平民消费级电脑单纯做推理都比较慢，更别说想自己从头开始训练一个模型了。 
本项目的目标是梳理整套生成式语言模型的训练流程，包括数据清洗、tokenizer训练、模型预训练、SFT指令微调、DPO偏好优化等。Chat-LM-small的模型参数只有0.7B，可以在最低16GB显存的机器训练（`fp16`或者` bf16`），推理最少只需要1GB显存（`bf16`，如果做`int8`、`int4`量化，还可以继续压缩）。 

- 公开所有预训练、SFT指令微调、DPO偏好优化数据集。
- 使用`Huggingface`NLP框架，包括`transformers`、`accelerate`、`trl`、`peft`等。
- 支持单机单卡、单机多卡进行训练、微调和推理。
- 预训练：整合为端到端的`text-to-text`预训练，非`mask`掩码预测预训练。
    - 开源所有数据清洗、数据集构造、数据集加载优化等流程；
    - tokenizer多进程词频统计，支持`sentencepiece`、`huggingface tokenizers`的tokenizer训练；
    - 预训练支持断点，可从断点处继续训练;
    - 大数据集（GB级别）流式加载、支持缓冲区数据打乱，配置`batch_size=16, max_len=320`下，最低支持在16GB内存+16GB显存的机器上进行训练；
    - 训练日志记录。
- SFT微调：prompt指令微调，低学习率，只训练decoder层。
    - SFT支持断点继续训练。
- 偏好优化：使用DPO进行全量偏好优化。
    - 支持使用peft lora进行偏好优化；
    - 支持模型合并，可将Lora adapter合并到原始模型中。


# 二、Chat-LM-small模型训练过程
## 2.1 数据集
所有数据集均来自互联网公开的**单轮对话**数据集，经过数据清洗、格式化后保存为parquet文件。数据处理过程见`utils/raw_data_process.py`。主要数据集包括： 

1. 社区问答json版webtext2019zh-大规模高质量数据集，见：[nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus)。共410万，清洗后剩余260万。
2. baike_qa2019百科类问答，见：<https://aistudio.baidu.com/datasetdetail/107726>，共140万，清醒后剩余130万。
3. 中国医药领域问答数据集，见：[Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data)，共79万，清洗后剩余79万。
4. ~~金融行业问答数据，见：<https://zhuanlan.zhihu.com/p/609821974>，共77万，清洗后剩余52万。~~**数据质量太差，未采用。**
5. 知乎问答数据，见：[Zhihu-KOL](https://huggingface.co/datasets/wangrui6/Zhihu-KOL)，共100万行，清洗后剩余97万行。
6. belle开源的指令训练数据，介绍：[BELLE](https://github.com/LianjiaTech/BELLE)，下载：[BelleGroup](https://huggingface.co/BelleGroup)，仅选取`Belle_open_source_1M`、`train_2M_CN`、及`train_3.5M_CN`中部分回答较短、不含复杂表格结构、翻译任务（没做英文词表）的数据，共370万行，清洗后剩余338万行。
7. 维基百科（Wikipedia）词条数据，将词条拼凑为提示语，百科的前`N`个词为回答，使用`202309`的百科数据，清洗后剩余119万的词条提示语和回答。Wiki下载：[zhwiki](https://dumps.wikimedia.org/zhwiki/)，将下载的bz2文件转换为wiki.txt参考：[WikiExtractor](https://github.com/apertium/WikiExtractor)。 

数据汇总：数据集总数量1023万：训练集：930万，评估集：2.5万（因为解码较慢，所以没有把评估集设置太大），测试集：90万。 

## 2.2 模型
T5模型（Text-To-Text Transfer Transformer），详情见论文: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)。

模型源码来自huggingface，见：[T5ForConditionalGeneration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1557)。

模型配置见`config.py`下的`T5ModelConfig`，官方的`T5-base`：`encoder layer`和`decoder layer `均为为12层，本项目这两个参数修改为10层。 

模型参数：0.7B。词表大小：29298，仅包含中文和少量英文。

## 2.3 训练过程
硬件：
```bash
CPU: 28 vCPU Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
内存：60 GB
显卡：RTX A5000(24GB) * 2
```
1. 预训练：学习率为`1e-4`到`5e-3`的动态学习率，训练时间为8天。训练损失：
![traing loss](img/train_loss.png)
2. prompt监督微调（SFT）：使用`belle`指令训练数据集（指令和回答长度都在320以下），学习率为`1e-5`到`1e-4`的动态学习率，冻结`encoder`参数，只微调`decoder`参数，微调时间1天。微调损失：
![finetune loss](img/finetune_loss.png)
3. dpo直接偏好优化：数据集[alpaca-gpt4-data-zh](https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh)作为`chosen`文本，步骤`2`中SFT模型对数据集中的prompt做批量`generate`，得到`rejected`文本，耗时1天，dpo全量偏好优化，学习率`le-5`，半精度`fp16`,共`2`个`epoch`，耗时2h。

## 2.4 对话效果展示
### 2.4.1 stream chat
`huggingface transformers`的 `TextIteratorStreamer`只支持`greedy search`，效果一般。
![](./img/stream_chat.gif)

### 2.4.2 chat
使用`beam search`，效果好于`greedy search`。
![](./img/chat.gif)

### 2.4.3 其他对话展示
![](./img/show1.png)
![](./img/show2.png)
![](./img/show3.png)
![](./img/show4.png)
![](./img/show5.png)
![](./img/show6.png)

存在问题：预训练数据集只有900多万，模型参数也仅0.7B，不能涵盖所有方面，会有答非所问、废话生成器的情况。

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
    
## 3.3 训练 
1. 训练数据集示例
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
    单机单卡：
    ``` bash
    accelerate launch ./train.py train
    ```
    单机多卡：
    ``` bash
    accelerate launch --multi_gpu --num_processes 2 ./train.py train
    ```
    从断点处继续训练：
    ```
    accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_keep_training=True
    ```

## 3.4 微调 
   
参考`data`目录下的示例`parquet`文件制作自己的数据集，数据集格式：`parquet`文件分两列，一列`prompt`文本，表示提示语，一列`response`文本，表示期待的模型输出。
微调细节见`model/trainer.py`下的`train`方法, `is_finetune`设置为`True`时，将进行微调，微调默认会冻结embedding层和encoder层，只训练decoder层。如需要冻结其他参数，请自行调整代码。 
**微调注意事项**: 学习率要低于`1e-5`，混合精度建议使用`fp16`，否则可能会出现`loss`为`Nan`的情况。

``` bash
accelerate launch --multi_gpu --num_processes 2 ./train.py --is_finetune=True
```

## 3.5 偏好优化
### 3.5.1 偏好优化方法
1.  使用RLHF（强化学习人类反馈优化方法，Reinforcement Learning from Human Feedback）做微调，既是PPO方法（近似偏好优化,Proximal Policy Optimization） 
    
    步骤1：使用微调数据集做有监督微调（SFT， Supervised Finetuning）。 
    步骤2：使用偏好数据集（一个prompt至少包含2个回复，一个想要的回复，一个不想要的回复。多个回复可以按照分数排序，最想要的分数最高）训练奖励模型（RM， Reward Model）。可使用`peft`库快速搭建Lora奖励模型。 
    步骤3：利用RM对SFT模型进行有监督PPO训练，使得模型满足偏好。 

2.  使用DPO（直接偏好优化，Direct Preference Optimization）微调（**本项目采用DPO微调方法，比较节省显存**）
    无需训练奖励模型，取得正向回答（chosen）和负向回答（rejected）即可开始微调。微调的`chosen`文本来自原数据集[alpaca-gpt4-data-zh](https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh)，拒绝文本`rejected`来自模型微调后的模型输出，处理过程见`utils/rlhf_data_process.py`。
### 3.5.2 偏好优化数据集示例
```json
{
    "prompt": "写一首关于假日的五行打油诗。",
    "chosen": "晴天阳光假期啊，\n遨游自在无虑夏。\n江河湖海尽畅游，\n感受自然人生华。\n假日时光真欢娱。",
    "reject": "假日的五行打油诗，\n五彩缤纷如画卷，\n春暖花开满眼眸，\n夏日炎炎暑气消。\n秋日炎炎烈日炙，\n冬日寒风凛冬雪，\n四季轮回尽收眼底，\n节日欢乐乐无边。\n这是我为您创作的一首五言绝句，希望您会喜欢。如果您还有其他需要我完成的任务，请随时告诉我。如果您有其他需要帮助的地方，我会尽力满足您的需求。请问还有什么可以帮到您的吗？如果您有任何其他需求，请随时与我联系。祝您愉快度过一个美好的假日。谢谢！如果您还有别的需求，欢迎随时联系我。祝您心情愉悦，假日快乐每一天。期待您的到来！再次感谢您的理解和支持！祝您假日愉快！愿您度过一个愉快的时光！祝您生活愉快！祝你好运！\n祝您新年快乐，万事如意！\n此致，敬礼\nXXX（注：以上为本人创作的内容，仅供参考）\n1.春春暖花开放，\n2.夏夏暑气消，\n3.秋秋寒风呼啸，\n4.冬冬雪飘零，\n5.新年佳节来临，\n6.春节欢声笑语欢。\n7.新年喜迎新春，\n8.国庆节快乐乐，\n9.感恩节快乐人，\n10.圣诞节快乐乐无穷。\n"
}
```
### 3.5.3 运行偏好优化
``` bash
python dpo_train.py
```

出现以下提示时，选择3即可
```bash
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 3
```

## 3.6 推理 
确保`model_save`目录下有以下文件：
```bash
chat_small_t5.best.dpo.bin
model_config.json
my_merged_tokenizer.json
```
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
    "input_txt": "感冒了要怎么办？"
}'
```
![api demo](./img/api_example.png)


# 四、引用
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

# 五、其他事项
本项目不承担开源模型和代码导致的数据安全、舆情风险或发生任何模型被误导、滥用、传播、不当利用而产生的风险和责任。

<!-- # 提示
```bash
# 导出项目依赖的包：
pipreqs --encoding "utf-8" --force
``` -->

