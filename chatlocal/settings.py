from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    by: str = "word"
    length: int
    language: str = "en"
    add_page: bool = True
    retrievertag: str
    top_k: int = 5
    docstorepath: Path
    embedding_dim: int = 768
    max_seq_length: int = 512
    api_key_string: str
    retriever_batch_size: int = 8


class Job(BaseModel):
    datadir: Path
    tag: str
