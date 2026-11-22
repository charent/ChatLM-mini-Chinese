"""数据集定义与构建。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset as HFDataset, load_dataset
from numpy import array, int64
from numpy.random import shuffle
from torch import LongTensor, cuda
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from chatlm.paths import DATA_DIR


class MyDataset(Dataset):
    """供 accelerate 手写训练循环使用的数据集。"""

    def __init__(
        self,
        parquet_file: str,
        tokenizer_dir: str,
        keep_in_memory: bool = False,
        max_seq_len: int = 512,
        buffer_size: int = 40960,
    ) -> None:
        """初始化。

        keep_in_memory: 是否将 parquet 文件转换为 pandas.DataFrame 格式存放到内存,
            False 将使用迭代生成器（迭代生成器不支持打乱全局数据），减少大数据集内存占用。
        """
        super().__init__()

        if cuda.device_count() >= 2 and not keep_in_memory:
            raise ValueError(
                "多GPU时使用MyDataset，参数keep_in_memory必须=True，否则无法进行分布式训练."
                f" 当前keep_in_memory={keep_in_memory}"
            )

        self.keep_in_memory = keep_in_memory
        self.max_seq_len = max_seq_len

        # 使用pyarrow.parquet读取，to_pandas、for遍历速度更快
        parquet_table = pq.read_table(parquet_file)

        # 获取数据集长度
        self.length = parquet_table.num_rows

        # 缓冲区大小不能超过数据长度
        self.buffer_size = self.length if buffer_size > self.length else buffer_size

        if keep_in_memory:
            # 转化为pandas放到内存中
            self.data = parquet_table.to_pandas()
        else:
            self.data = parquet_table

        # 初始化tokenizer
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

        # 在这里初始化generator
        self.sample_generator = self.item_generator()

    def item_generator(self) -> Iterator[tuple[str, str]]:
        """一条数据的生成器，防止大数据集 OOM。"""
        parquet_table = self.data

        # 生成器是死循环，不用退出，训练结束（epoch结束）会停止调用 next()
        buffer_list = []
        while True:
            for prompt, response in zip(parquet_table["prompt"], parquet_table["response"]):
                # 缓存数据不够，添加数据
                if len(buffer_list) < self.buffer_size:
                    buffer_list.append((prompt.as_py(), response.as_py()))
                    continue

                # 执行到这里，缓存区够了，打乱数据
                shuffle(buffer_list)
                for p, r in buffer_list:
                    yield p, r

                # 迭代完成，清空缓存区
                buffer_list = []

    def __getitem__(self, index: int) -> tuple[str, str]:
        """返回一条样本。"""
        if self.keep_in_memory:
            data = self.data
            prompt, response = data.iloc[index].prompt, data.iloc[index].response
        else:
            prompt, response = next(self.sample_generator)

        max_seq_len = self.max_seq_len - 5  # len('[EOS]') = 5
        # add an eos token note that end of resopnse, using in generate.
        return f"{prompt[0: max_seq_len]}[EOS]", f"{response[0: max_seq_len]}[EOS]"

    def collate_fn(self, data: list[list]) -> dict:
        """合并一个批次数据返回。"""
        tokenizer = self.tokenizer

        prompt = tokenizer([item[0] for item in data], padding=True, return_token_type_ids=False)
        response = tokenizer([item[1] for item in data], padding=True, return_token_type_ids=False)

        input_ids = array(prompt.input_ids, dtype=int64)
        input_mask = array(prompt.attention_mask, dtype=int64)
        target_ids = array(response.input_ids, dtype=int64)

        return {
            "input_ids": LongTensor(input_ids),
            "input_mask": LongTensor(input_mask),
            "target_ids": LongTensor(target_ids),
        }

    def __len__(self) -> int:
        return self.length


def build_seq2seq_dataset(
    file: str,
    split: str,
    tokenizer: PreTrainedTokenizerFast,
    file_format: str = "json",
    cache_dir: str | None = None,
) -> HFDataset:
    """加载并编码序列到序列训练数据集。

    替代 pre_train/sft_train/dpo_train 中重复的 ``get_dataset`` + ``tokens_to_ids``。

    参数:
        file: 数据文件路径
        split: 数据集切分名称，如 ``train``
        tokenizer: 已加载的 tokenizer
        file_format: ``json`` 或 ``parquet``
        cache_dir: datasets 缓存目录，默认使用 HuggingFace 默认缓存
    """
    # load_dataset 在传入 split 时返回 Dataset，但存根声明为四类联合类型
    dataset = cast(HFDataset, load_dataset(file_format, data_files=file, split=split, cache_dir=cache_dir))
    column_names: list[str] = list(dataset.column_names)

    def tokens_to_ids(samples: dict[str, Any]) -> dict[str, list[Any]]:
        eos_token_id = tokenizer.eos_token_id

        encoded_prompt = tokenizer(samples["prompt"], truncation=False, padding=False, return_attention_mask=False)
        encoded_response = tokenizer(samples["response"], truncation=False, padding=False, return_attention_mask=False)

        # vocab size 小于65535 可以用 uint16, 每个样本都要添加eos_token_id
        input_ids = [np.array(item + [eos_token_id], dtype=np.uint16) for item in encoded_prompt.input_ids]
        labels = [np.array(item + [eos_token_id], dtype=np.uint16) for item in encoded_response.input_ids]

        return {"input_ids": input_ids, "labels": labels}

    return cast(
        HFDataset,
        dataset.map(tokens_to_ids, batched=True, batch_size=8192, remove_columns=column_names),
    )


def main() -> None:
    """数据集冒烟测试。

    >>> python -m chatlm.data.dataset
    """
    from torch.utils.data import DataLoader

    parquet_file = str(DATA_DIR / "my_valid_dataset_1k.parquet")
    tokenizer_dir = str(DATA_DIR.parent / "artifacts")

    dataset = MyDataset(parquet_file, tokenizer_dir, keep_in_memory=False, max_seq_len=128)
    print("dataset size: ", len(dataset))
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)

    for epoch in range(2):
        print(f"epoch: {epoch}")
        for step, batch in enumerate(dataloader):
            x, x_mask, y = batch["input_ids"], batch["input_mask"], batch["target_ids"]
            print(f"step:{step}", x.shape, x_mask.shape, y.shape)
            if step == 5:
                break


if __name__ == "__main__":
    main()


__all__ = ["MyDataset", "build_seq2seq_dataset"]
