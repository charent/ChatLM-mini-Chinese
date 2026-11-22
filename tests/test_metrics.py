"""指标计算测试。"""

from chatlm.utils.metrics import extract_Ngram, get_bleu4_score, my_average


def test_my_average_empty() -> None:
    assert my_average([]) == 0.0


def test_my_average_values() -> None:
    assert my_average([1.0, 2.0, 3.0]) == 2.0


def test_extract_ngram_counts() -> None:
    counter = extract_Ngram(["a", "b", "c"], 2)

    assert counter[("a", 1)] == 1
    assert counter[("a b", 2)] == 1
    assert counter[("b c", 2)] == 1


def test_bleu4_identical_sentence() -> None:
    assert get_bleu4_score("你好世界", "你好世界") > 0.99


def test_bleu4_disjoint_sentence() -> None:
    assert get_bleu4_score("abcd", "wxyz") == 0.0
