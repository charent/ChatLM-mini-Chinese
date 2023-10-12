# Chat-LM-small

### 一、介绍
*Read this in [English](README.en.md).*

### 二、模型
##### 2.1 数据集
所有数据集均来自互联网公开的**单轮对话**数据集，经过数据清洗、格式化后保存为parquet文件。数据处理过程见`utils/raw_data_process.py`。主要数据集包括： 

1. 社区问答json版webtext2019zh-大规模高质量数据集，见：<https://github.com/brightmart/nlp_chinese_corpus>。共410万，清洗后剩余260万。
2. baike_qa2019百科类问答，见：<https://aistudio.baidu.com/datasetdetail/107726>，共140万，清醒后剩余130万。
3. 中国医药领域问答数据集，见：<https://github.com/Toyhom/Chinese-medical-dialogue-data>，共79万，清洗后剩余79万。
4. ~~金融行业问答数据，见：<https://zhuanlan.zhihu.com/p/609821974>，共77万，清洗后剩余52万。~~**数据质量太差，未采用。**
5. 知乎问答数据，见：<https://huggingface.co/datasets/wangrui6/Zhihu-KOL>，共100万行，清洗后剩余97万行。
6. belle开源的知识增强数据集，介绍：<https://github.com/LianjiaTech/BELLE>，下载：<https://huggingface.co/BelleGroup>，仅选取`Belle_open_source_1M`、`train_2M_CN`、及`train_3.5M_CN`中部分回答较短、不含复杂表格结构、翻译任务（没做英文词表）的数据，共370万行，清洗后剩余338万行。
7. 维基百科（Wikipedia）词条数据，将词条拼凑为提示语，百科的前`N`个词为回答，使用202309的百科数据，清洗后剩余119万的词条提示语和回答。Wiki下载：<https://dumps.wikimedia.org/zhwiki/>，将下载的bz2文件转换为wiki.txt参考：<https://github.com/apertium/WikiExtractor>。 

数据汇总：数据集总数量1023万：训练集：930万，评估集：2.5万（因为解码较慢，所以没有把评估集设置太大），测试集：90万。 

#### 2.2 模型


#### 2.3 训练过程


#### 2.4 效果展示


### 三、使用说明

1. 安装依赖 
    pip安装：
    ```bash
    pip install -r ./requirements.txt
    ``` 
    conda安装：
    ```bash
    conda install --yes --file ./requirements.txt
    ```
    
2. 训练
   
    2.1 jupyter-lab 或者 jupyter notebook: 
    见文件`train.ipynb`，推荐使用jupyter-lab，避免考虑与服务器断开后终端进程被杀的情况。 

    2.2 终端：
    终端训练需要考虑连接断开后进程被杀的，推荐使用进程守护工具`Supervisor`或者`screen`建立连接会话。

    首先要配置`accelerate`，执行以下命令， 根据提示选择即可，参考`accelerate.yaml`，*注意：DeepSpeed在Windows安装比较麻烦*。
    ``` bash
    accelerate config
    ```
    开始训练，如果要使用工程提供的配置请在下面的命令`accelerate launch`后加上参数`--config_file ./accelerate.yaml`，*该配置按照单机2xGPU配置*
    单机单卡：
    ``` bash
    accelerate launch ./train.py train
    ```
    单机多卡：
    ``` bash
    accelerate launch --multi_gpu --num_processes 2 ./train.py train
    ```

3.  微调
   见`model/trainer.py`下的`finetrun`方法
    ``` bash
    accelerate launch --multi_gpu --num_processes 2 ./train.py finetrune=True
    ```
4.  推理
    控制台运行：
    ```bash
    python cli_demo.py
    ```


#### 提示

```bash
# 导出项目依赖的包：
pipreqs --encoding "utf-8" --force

# pip 安装torch + cu118
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

