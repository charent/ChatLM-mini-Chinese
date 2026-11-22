"""文本清洗与去重工具。"""

from __future__ import annotations

import re
from collections import defaultdict

from datasketch import MinHash, MinHashLSH

# 结束标点符号
END_PUN = set('.。!！）)》}】?？"”')

# 保留中文和英文、下划线，不要标点符号
NON_CHAR = re.compile("[^[\u4e00-\u9fa5|A-Za-z_0-9]")


def _get_doc_mini_hash(doc: list[str] | str, num_perm: int) -> MinHash:
    """获取一段文本的 mini hash。"""
    mini_hash = MinHash(num_perm=num_perm)
    for s in doc:
        mini_hash.update(s.encode("utf-8"))
    return mini_hash


class DropDatasetDuplicate:
    """基于 MinHash LSH 的近似去重。"""

    def __init__(self, threshold: float = 0.85, num_perm: int = 256) -> None:
        """获取一个数据集中所有重复（相似的超过 threshold）的 index。

        输入为 list[str]，一个 str 元素为一段文本（doc）。
        如输入 [a, b, c, d, c, d, e] 返回 {4, 5}（后面两个 c, d 的 index）。
        """
        self.similar_index_cluster: dict[int, set[int]] = defaultdict(set)
        self.data_lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm

    def add_doc(self, index: object, doc: str) -> None:
        """添加文档。

        index: 文档的索引
        doc: 文档本身
        """
        # 只保留中文和英文、下划线，不要标点符号
        doc = "".join(NON_CHAR.split(doc))

        doc_hash = _get_doc_mini_hash(doc, self.num_perm)
        close_duplicates = self.data_lsh.query(doc_hash)

        self.data_lsh.insert(index, doc_hash)

        # 所有相似的 doc 在 similar_index_cluster 中的 key 都是最早出现的 idx
        if len(close_duplicates) > 0:
            min_idx = min(close_duplicates)
            self.similar_index_cluster[min_idx].add(index)

    def get_duplicate_indexs(self) -> set[int]:
        """返回所有的重复文档索引。"""
        similar_index_cluster = self.similar_index_cluster
        need_to_remove_idx: set[int] = set()

        for key_idx in similar_index_cluster.keys():
            need_to_remove_idx |= similar_index_cluster[key_idx]

        return need_to_remove_idx


def fixed_response(item: str) -> str:
    """修复被截断的回答，从末尾往回找第一个结束标点。"""
    if len(item) <= 1:
        return item
    if item[-1] in END_PUN:
        return item

    n = len(item)
    i = n - 1
    while i > 0 and item[i] not in END_PUN:
        i -= 1

    return "".join(item[0 : i + 1])


def fixed_space(sentence: str) -> str:
    """单个空格删除，连续两个空格保留一个。"""
    n = len(sentence)
    new_sentence = []
    i = 0
    while i < n:
        word = sentence[i]
        if word != " ":
            new_sentence.append(word)
        elif i + 1 < n and sentence[i + 1] == " ":
            new_sentence.append(word)
            i += 1  # 两个空格保留一个，指针往下走一步
        i += 1

    return "".join(new_sentence)


__all__ = ["END_PUN", "NON_CHAR", "DropDatasetDuplicate", "fixed_response", "fixed_space"]
