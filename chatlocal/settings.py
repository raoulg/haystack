from pathlib import Path

import numpy as np
import tomli
from loguru import logger
from pydantic import BaseModel


class UserInput(BaseModel):
    datadir: str
    tag: str
    top_k: int
    length: int
    retrievertag: str
    embedding_dim: int
    max_seq_length: int
    extractor_K: int
    blocks: int
    main_topic: str
    keyword_n: int
    textgraph_K: int

    @classmethod
    def fromtoml(cls, tomlsettings):
        return cls(
            datadir=tomlsettings["job"]["datadir"],
            tag=tomlsettings["job"]["tag"],
            top_k=tomlsettings["job"]["top_k"],
            length=tomlsettings["builder"]["length"],
            retrievertag=tomlsettings["builder"]["retrievertag"],
            embedding_dim=tomlsettings["builder"]["embedding_dim"],
            max_seq_length=tomlsettings["builder"]["max_seq_length"],
            extractor_K=tomlsettings["extractor"]["K"],
            blocks=tomlsettings["extractor"]["blocks"],
            textgraph_K=tomlsettings["textgraph"]["K"],
            main_topic=tomlsettings["extractor"]["main_topic"],
            keyword_n=tomlsettings["keywords"]["n"],
        )


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


def load_userinput(tomlfile: Path) -> UserInput:
    if not tomlfile.exists():
        msg = f"no chatlocal.toml file found in {Path.cwd()}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    else:
        logger.info(f"loading settings from {tomlfile}")
        with tomlfile.open("rb") as f:
            tomlsettings = tomli.load(f)
            userinput = UserInput.fromtoml(tomlsettings)
        logger.info("=== job settings ===")
        logger.info(f"running with tag {userinput.tag} and top_k {userinput.top_k}")
        logger.info("=== builder settings ===")
        logger.info(
            f"using retrievertag {userinput.retrievertag} with length {userinput.length}"
        )
        logger.info(
            f"using embed_dim {userinput.embedding_dim} and max_seq_length {userinput.max_seq_length}"
        )
        logger.info("=== extractor settings ===")
        logger.info(
            f"using K={userinput.extractor_K} and blocks={userinput.blocks}, topic {userinput.main_topic}"
        )
    return userinput
