"""评估指标：BLEU4、F1/精确率/召回率、均值。"""

from __future__ import annotations

from collections import Counter

import numpy as np


def my_average(arry_list: list[float]) -> float:
    """自定义均值计算，空数组返回 0.0。"""
    if len(arry_list) == 0:
        return 0.0

    return float(np.average(arry_list))


def extract_Ngram(words_list: list[str], n_gram: int) -> Counter:
    """获取一个句子的 n_gram。

    return:
        ngram_counter: key = ('w1 w2 ... wn', n_gram)，value 为出现次数
    """
    n = len(words_list)
    ngram_counter: Counter = Counter()

    for i in range(1, n_gram + 1):
        for j in range(n - i + 1):
            key = " ".join(words_list[j : j + i])
            ngram_counter[(key, i)] += 1

    return ngram_counter


def get_bleu4_score(
    reference: str | list[str],
    outputs: str | list[str],
    n_gram: int = 4,
) -> float:
    """获取 bleu4 分数。"""
    weights = np.ones(n_gram) * (1.0 / n_gram)

    outputs_len, reference_len = len(outputs), len(reference)

    if not isinstance(reference, list):
        reference = list(reference)
    if not isinstance(outputs, list):
        outputs = list(outputs)

    outputs_counter = extract_Ngram(outputs, n_gram=n_gram)
    reference_counter = extract_Ngram(reference, n_gram=n_gram)

    ngram_counter_clip = outputs_counter & reference_counter

    clip_counter = np.zeros(n_gram)
    output_ngram_counter = np.zeros(n_gram)

    for (_key, ngram), cnt in ngram_counter_clip.items():
        clip_counter[ngram - 1] += cnt

    for (_key, ngram), cnt in outputs_counter.items():
        output_ngram_counter[ngram - 1] += cnt

    if np.min(clip_counter) == 0.0:
        return float(np.array(0.0))

    precision_scores = clip_counter / output_ngram_counter

    # bleu
    log_precision_scores = weights * np.log(precision_scores)

    # 几何平均形式求平均值然后加权
    geometric_mean = np.exp(np.sum(log_precision_scores))
    brevity_penalty = np.exp(1 - (reference_len / outputs_len))

    bleu = brevity_penalty * geometric_mean

    return float(bleu)


def f1_p_r_compute(spo_list_pred: list, spo_list_true: list, repair: bool = False):
    """计算 spo 的 f1 分数、精确率、召回率。

    spo_list: [[(s,p,o)...], [(s,p,o)]]，每一行 [(s,p,o)...] 为一个句子中的 spo。
    """
    assert len(spo_list_pred) == len(spo_list_true)

    def repair_song_album(spo_list: list, song: list, album: list):
        """修复一条文本的 '歌曲' 和 '专辑' 的 spo。

        对于歌曲 x（subject）的关系歌手、作词、作曲，x 必须同时存在于 song 和 album 中。
        """
        if len(song) == 0 and len(album) == 0:
            return spo_list

        ps = ["歌手", "作词", "作曲"]
        new_spo_list = []
        for spo in spo_list:
            s, p = spo[0], spo[1]
            if p in ps and s in album and s not in song:
                continue
            new_spo_list.append(spo)

        return new_spo_list

    def repair_song_album_list(spo_list: list):
        new_spo_list = []
        for spos in spo_list:
            song, album = [], []
            for spo in spos:
                s, p, o = spo
                if p == "所属专辑":
                    song.append(s)
                    album.append(o)
            new_spo_list.append(repair_song_album(spos, song, album))

        return new_spo_list

    if repair:
        spo_list_pred = repair_song_album_list(spo_list_pred)
        spo_list_true = repair_song_album_list(spo_list_true)

    TP = 1e-10  # 正类判定为正类, A
    TP_FP = 1e-10  # 检索到的, A + B
    TP_FN = 1e-10  # 真正想要的，A + C

    for i in range(len(spo_list_true)):
        pred_set = set(spo_list_pred[i])
        true_set = set(spo_list_true[i])

        pred_true_set = pred_set & true_set  # 预测和真实取交集

        TP += len(pred_true_set)  # 检索到且是想要的， A
        TP_FP += len(pred_set)  # 检索到的，包括想要的和不想要的，A + B
        TP_FN += len(true_set)  # 真正想要的， 包括检索到和没检索到的，A + C

    p = TP / TP_FP
    r = TP / TP_FN
    f1 = (2 * p * r) / (p + r)

    return f1, p, r


__all__ = ["my_average", "extract_Ngram", "get_bleu4_score", "f1_p_r_compute"]
