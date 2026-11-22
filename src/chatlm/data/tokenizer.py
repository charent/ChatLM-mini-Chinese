"""Tokenizer 训练：HuggingFace BPE 与 sentencepiece BPE。

用法::

    chatlm-tokenizer --corpus-file data/wiki.simple.txt --token-type char
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator

import sentencepiece as spm
import tokenizers
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Digits, Metaspace, Punctuation
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from chatlm.paths import ARTIFACT_DIR, DATA_DIR

SPECIAL_TOKENS = ["[PAD]", "[EOS]", "[SEP]", "[BOS]", "[CLS]", "[MASK]", "[UNK]"]


def ensure_dir(directory: str) -> None:
    """检查文件夹是否存在，如果不存在则创建。"""
    from pathlib import Path

    Path(directory).mkdir(parents=True, exist_ok=True)


def train_my_huggingface_wiki_tokenizer(
    corpus_file: str,
    max_train_line: int | None = None,
    vocab_size: int = 40960,
    token_type: str = "char",
) -> None:
    """训练 tokenizer with huggingface，至少需要 32G 内存，运行大概需要半个小时。"""
    tokenizer_slow_save_path = ARTIFACT_DIR / "hf_tokenizer_slow" / "hf_bpe_tokenizer.json"
    tokenizer_fast_save_dir = ARTIFACT_DIR

    ensure_dir(str(tokenizer_slow_save_path.parent))
    ensure_dir(str(tokenizer_fast_save_dir))

    def get_training_corpus(buffer_size: int = 1000, chunk_len: int = 2048) -> Iterator[list[str]]:
        """一个文本块大小 2048。"""
        line_cnt = 0
        buffer: list[str] = []
        with open(corpus_file, "r", encoding="utf-8") as f_read:
            cur_chunk_txt, txt_len = [], 0
            for line in f_read:
                cur_chunk_txt.append(line)
                txt_len += len(line)
                line_cnt += 1

                if txt_len >= chunk_len:
                    buffer.append("".join(cur_chunk_txt))
                    cur_chunk_txt, txt_len = [], 0

                if len(buffer) >= buffer_size:
                    yield buffer
                    buffer = []

                if isinstance(max_train_line, int) and line_cnt > max_train_line:
                    break

            # yield last
            if len(buffer) > 0:
                yield buffer

    special_tokens = SPECIAL_TOKENS

    if token_type == "char":
        model = BPE(unk_token="[UNK]")
        tokenizer = Tokenizer(model)

        # 用兼容等价分解合并对utf编码进行等价组合，比如全角A转换为半角A
        tokenizer.normalizer = tokenizers.normalizers.Sequence([NFKC()])

        # 标点符号，数字，及Metaspace预分割（否则decode出来没有空格）
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([Punctuation(), Digits(individual_digits=True), Metaspace()])

        tokenizer.add_special_tokens(special_tokens)
        tokenizer.decoder = decoders.Metaspace()
    elif token_type == "byte":
        # byte BPE 不需要 unk_token
        model = BPE()
        tokenizer = Tokenizer(model)
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)

        tokenizer.add_special_tokens(special_tokens)
        tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.post_processor = tokenizers.processors.ByteLevel(trim_offsets=False)
    else:
        raise ValueError(f"token type must be `char` or `byte`, but got {token_type}")

    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=100, show_progress=True, special_tokens=special_tokens)
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # add \t \n
    if "\t" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["\t"])
    if "\n" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["\n"])

    tokenizer.save(str(tokenizer_slow_save_path))

    # 将训练的tokenizer转换为PreTrainedTokenizerFast并保存
    # 转换是为了方便作为`AutoTokenizer`传到其他`huggingface`组件使用。
    # 转换时要手动指定`pad_token`、`eos_token`等特殊token
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )

    fast_tokenizer.save_pretrained(str(tokenizer_fast_save_dir))

    print(f"slow tokenizer save in path: {tokenizer_slow_save_path}")
    print(f"fast tokenizer save in path: {tokenizer_fast_save_dir}")
    print(f"\ntrain tokenizer finished. use `AutoTokenizer.from_pretrained('{tokenizer_fast_save_dir}')` to load and test your tokenizer.")


def train_my_BPE_tokenizer(corpus_file: str | None = None, vocab_size: int = 40960) -> None:
    """使用 sentencepiece 训练 BPE，缺点只能加载 300 万行，16G 内存会 OOM。"""
    txt_corpus_file = corpus_file or str(DATA_DIR / "my_corpus.txt")
    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]

    spm.SentencePieceTrainer.train(
        input=txt_corpus_file,
        model_prefix=str(ARTIFACT_DIR / "my_tokenizer"),
        vocab_size=vocab_size,
        user_defined_symbols=special_tokens,
        max_sentence_length=1024,
        shuffle_input_sentence=True,
        model_type="bpe",
    )

    # 模型文件保存为 artifacts/my_tokenizer.model


def main() -> None:
    parser = argparse.ArgumentParser(description="训练项目 tokenizer")
    parser.add_argument("--backend", choices=["hf", "sentencepiece"], default="hf", help="训练后端")
    parser.add_argument("--corpus-file", default=str(DATA_DIR / "wiki.simple.txt"), help="训练语料文件")
    parser.add_argument("--vocab-size", type=int, default=40960, help="词表大小")
    parser.add_argument("--token-type", choices=["char", "byte"], default="char", help="BPE 类型（仅 hf 后端）")
    parser.add_argument("--max-train-line", type=int, default=None, help="最多训练的行数（仅 hf 后端）")
    args = parser.parse_args()

    if args.backend == "hf":
        train_my_huggingface_wiki_tokenizer(
            corpus_file=args.corpus_file,
            max_train_line=args.max_train_line,
            vocab_size=args.vocab_size,
            token_type=args.token_type,
        )
    else:
        train_my_BPE_tokenizer(corpus_file=args.corpus_file, vocab_size=args.vocab_size)


if __name__ == "__main__":
    main()


__all__ = ["train_my_huggingface_wiki_tokenizer", "train_my_BPE_tokenizer", "main"]
