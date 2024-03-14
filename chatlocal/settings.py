from pathlib import Path

import numpy as np
from pydantic import BaseModel


class Settings(BaseModel):
    by: str = "word"
    length: int
    language: str = "en"
    add_page: bool = True
    retrievertag: str
    docstorepath: Path = Path.home() / ".cache/chatlocal/"
    embedding_dim: int = 768
    max_seq_length: int = 512
    api_key_string: str
    retriever_batch_size: int = 8


class Job(BaseModel):
    datadir: Path
    tag: str
    questionsfile: Path
    top_k: int


class Embeddings(BaseModel):
    v: np.ndarray
    ids: list

    class Config:
        arbitrary_types_allowed = True


class Clusters(BaseModel):
    center: dict
    random: dict

    def __repr__(self):
        k1 = len(self.center)
        v1 = len(self.center[0])
        k2 = len(self.random)
        v2 = len(self.random[0])

        return f"Clusters({k1}x{v1} center, {k2}x{v2} random)"


class ExtractorSettings(BaseModel):
    K: int
    blocks: int
    spectral_extradims: int


class PromtOptions(BaseModel):
    l1: int
    l2: int
    docfun: str
    main_topic: str
    avoid: set
    keywordmodel: str
