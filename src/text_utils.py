"""torchtext 已被官方棄用且與新版 PyTorch 不相容，
此模組提供 torchtext.data.utils / torchtext.vocab / torchtext.functional
中，本書範例實際用到的功能的輕量替代實作。
"""

import os
import re
import urllib.request
import zipfile
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
from torch.nn.utils.rnn import pad_sequence

_BASIC_ENGLISH_PATTERNS = [
    (re.compile(r"\'"), " '  "),
    (re.compile(r"\""), ""),
    (re.compile(r"\."), " . "),
    (re.compile(r"<br \/>"), " "),
    (re.compile(r","), " , "),
    (re.compile(r"\("), " ( "),
    (re.compile(r"\)"), " ) "),
    (re.compile(r"\!"), " ! "),
    (re.compile(r"\?"), " ? "),
    (re.compile(r"\;"), " "),
    (re.compile(r"\:"), " "),
    (re.compile(r"\s+"), " "),
]


def _basic_english_normalize(line: str) -> List[str]:
    line = line.lower()
    for pattern_re, replaced_str in _BASIC_ENGLISH_PATTERNS:
        line = pattern_re.sub(replaced_str, line)
    return line.split()


def get_tokenizer(tokenizer: Optional[str] = "basic_english") -> Callable[[str], List[str]]:
    if tokenizer is None:
        return str.split
    if tokenizer == "basic_english":
        return _basic_english_normalize
    raise ValueError(f"不支援的 tokenizer：{tokenizer}")


class Vocab:
    """對應 torchtext.vocab.Vocab 中，本書範例用到的介面子集。"""

    def __init__(self, itos: List[str]) -> None:
        self.itos = list(itos)
        self.stoi: Dict[str, int] = {token: i for i, token in enumerate(self.itos)}
        self.default_index: Optional[int] = None

    def set_default_index(self, index: int) -> None:
        self.default_index = index

    def __getitem__(self, token: str) -> int:
        if token in self.stoi:
            return self.stoi[token]
        if self.default_index is not None:
            return self.default_index
        raise RuntimeError(f"詞彙表中查無此單字，且未設定 default index：{token}")

    def __len__(self) -> int:
        return len(self.itos)

    def get_itos(self) -> List[str]:
        return self.itos

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return [self[token] for token in tokens]


def vocab(ordered_dict: Dict[str, int], specials: Optional[List[str]] = None) -> Vocab:
    """對應 torchtext.vocab.vocab 工廠函數。"""
    specials = specials or []
    tokens = [token for token in ordered_dict if token not in specials]
    itos = list(specials) + tokens
    return Vocab(itos)


def truncate(input: List[List[int]], max_seq_len: int) -> List[List[int]]:
    return [ids[:max_seq_len] for ids in input]


def to_tensor(input: List[List[int]], padding_value: Optional[int] = None) -> torch.Tensor:
    if padding_value is None:
        return torch.tensor(input, dtype=torch.long)
    return pad_sequence(
        [torch.tensor(ids, dtype=torch.long) for ids in input], batch_first=True, padding_value=float(padding_value)
    )


class GloVe:
    """從 GloVe 官方 txt 檔載入詞向量，取代 torchtext.vocab.GloVe。"""

    _URLS = {"6B": "https://nlp.stanford.edu/data/glove.6B.zip"}

    def __init__(self, name: str = "6B", dim: int = 50, cache: str = "glove") -> None:
        self.dim = dim
        os.makedirs(cache, exist_ok=True)
        txt_path = os.path.join(cache, f"glove.{name}.{dim}d.txt")

        if not os.path.exists(txt_path):
            self._download(name, cache)

        itos: List[str] = []
        vectors: List[List[float]] = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                itos.append(values[0])
                vectors.append([float(v) for v in values[1:]])

        self.itos = itos
        self.stoi: Dict[str, int] = {word: i for i, word in enumerate(itos)}
        self.vectors = torch.tensor(vectors, dtype=torch.float32)

    def _download(self, name: str, cache: str) -> None:
        if name not in self._URLS:
            raise ValueError(f"不支援的 GloVe 版本：{name}")

        zip_path = os.path.join(cache, f"glove.{name}.zip")
        print(f"下載 GloVe 詞向量檔：{self._URLS[name]}")
        urllib.request.urlretrieve(self._URLS[name], zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache)

        os.remove(zip_path)

    def __getitem__(self, token: str) -> torch.Tensor:
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        return torch.zeros(self.dim)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def get_vecs_by_tokens(self, tokens: Union[str, Sequence[str]], lower_case_backup: bool = False) -> torch.Tensor:
        to_reduce = False
        if isinstance(tokens, str):
            tokens = [tokens]
            to_reduce = True

        if not lower_case_backup:
            vecs = torch.stack([self[token] for token in tokens])
        else:
            vecs = torch.stack([self[token] if token in self.stoi else self[token.lower()] for token in tokens])

        return vecs[0] if to_reduce else vecs
