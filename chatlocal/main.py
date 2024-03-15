import os
import pickle
import sys
from pathlib import Path

import tomli
from loguru import logger
from models import get_embeddings

from chatlocal.builder import DocumentstoreBuilder
from chatlocal.graphs import Build_kw_weights, save_weights, sparse_weightmatrix
from chatlocal.llm import KeywordExtraction
from chatlocal.models import ExtractClusters
from chatlocal.settings import ExtractorSettings, Job, PromtOptions, Settings

logger.remove()
logger.add("logs/logfile.log", level="DEBUG")
logger.add(sys.stderr, level="INFO")


def main(skip_add: bool = True):
    tomlfile = Path("chatlocal.toml")
    if not tomlfile.exists():
        logger.info(f"no chatlocal.toml file found in {Path.cwd()}")
    else:
        logger.info(f"loading settings from {tomlfile}")
        with tomlfile.open("rb") as f:
            tomlsettings = tomli.load(f)

    # loading settings from toml
    tag = tomlsettings["job"]["tag"]
    top_k = tomlsettings["job"]["top_k"]
    logger.info("=== job settings ===")
    logger.info(f"running with tag {tag} and top_k {top_k}")
    buildset = tomlsettings["builder"]
    retrievertag, length = buildset["retrievertag"], buildset["length"]
    embed_dim, max_seq_length = buildset["embedding_dim"], buildset["max_seq_length"]
    logger.info("=== builder settings ===")
    logger.info(f"using retrievertag {retrievertag} with length {length}")
    logger.info(f"using embed_dim {embed_dim} and max_seq_length {max_seq_length}")
    K = tomlsettings["extractor"]["K"]
    blocks = tomlsettings["extractor"]["blocks"]
    logger.info("=== extractor settings ===")
    logger.info(f"using K={K} and blocks={blocks}")

    job = Job(
        datadir=Path.home() / "Downloads/research",
        tag=tag,
        questionsfile=Path.home() / "code/haystack/dev/questions.txt",
        top_k=top_k,
    )
    apikeystring = "OPENAI_API_KEY"
    API_KEY = os.environ.get(apikeystring, None)

    settings = Settings(
        by="word",
        length=length,
        docstorepath=Path.home() / ".cache/chatlocal",
        retrievertag=retrievertag,
        embedding_dim=embed_dim,
        max_seq_length=max_seq_length,
        api_key_string=API_KEY,
    )

    # get documentstore
    builder = DocumentstoreBuilder(settings=settings)
    if not skip_add:
        logger.info(f"adding files from {job.datadir} to the docstore")
        builder.add_files(job=job)

    docstore = builder.get_docstore(job=job)
    # get embeddings
    emb = get_embeddings(docstore)

    # extract clusters
    extractorsettings = ExtractorSettings(K=K, blocks=blocks, spectral_extradims=5)
    extractor = ExtractClusters(emb, extractorsettings)
    clusters = extractor(clusterfile=Path("artefacts/clusters.pkl"))

    # extract keywords
    promptoptions = PromtOptions(
        l1=10,
        l2=3,
        docfun="{' * '.join([d.content for d in documents])}",
        main_topic="energy poverty",
        avoid=set(),
        keywordmodel="gpt-3.5-turbo",
    )

    kwextractor = KeywordExtraction(options=promptoptions, apikey=apikeystring)
    keywordsfile = Path("artefacts/keywords.txt")

    keywords = kwextractor(clusters, builder, keywordsfile)
    print(f"extracted {len(keywords)} keywords to {keywordsfile}")

    textweightsfile = Path("artefacts/textweights.pkl")
    if not textweightsfile.exists():
        Wt = sparse_weightmatrix(K=30, emb=emb)
        logger.info(f"saving textweights to {textweightsfile}")
        save_weights(Wt, Path("artefacts/textweights.pkl"))
    else:
        logger.info(f"found {textweightsfile}, skipping textweights creation")
        with textweightsfile.open("rb") as f:
            Wt = pickle.load(f)

    embeddingsfile = Path("artefacts/kw2textassociation.npy")
    kw_weightsfile = Path("artefacts/kw_weights.pkl")

    kwgraph = Build_kw_weights(
        keywordsfile, embeddingsfile, kw_weightsfile, apikeystring
    )
    kw_weights = kwgraph(emb, Wt)
    logger.info(f"Created kw_weights with {kw_weights.keys()}")


if __name__ == "__main__":
    main()
