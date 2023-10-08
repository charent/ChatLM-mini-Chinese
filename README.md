# Chat-LM-small

#### 介绍
*Read this in [English](README.en.md).*

#### 软件架构
软件架构说明


#### 安装教程

1.  xxxx
2.  xxxx
3.  xxxx

#### 使用说明

1.  运行
``` bash
# 单卡
accelerate launch ./main.py train

# 多卡
accelerate launch --multi_gpu --num_processes 2 ./main.py train
```
   
2.  xxxx
3.  xxxx


#### tips

```bash
# 导出项目依赖的包：
pipreqs --encoding "utf-8" --force

# pip 安装torch + cu118
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

