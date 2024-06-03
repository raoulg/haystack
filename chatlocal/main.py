import os
import pickle
import sys
from pathlib import Path

from loguru import logger
from chatlocal.models import get_embeddings

from chatlocal.builder import DocumentstoreBuilder
from chatlocal.graphs import Build_kw_weights, save_weights, sparse_weightmatrix
from chatlocal.llm import KeywordExtraction, keywordcleanup
from chatlocal.models import ExtractClusters
from chatlocal.settings import (
    ExtractorSettings,
    Job,
    PromtOptions,
    Settings,
    load_userinput,
)

logger.remove()
logger.add("logs/logfile.log", level="DEBUG")
logger.add(sys.stderr, level="INFO")


def main(update_db: bool = False, new_database: bool = False):
    tomlfile = Path("chatlocal.toml")
    userinput = load_userinput(tomlfile)
    outputdir = Path(f"artefacts/{userinput.tag}")
    if not outputdir.exists():
        outputdir.mkdir(parents=True)

    job = Job(
        datadir=Path.home() / userinput.datadir,
        tag=userinput.tag,
        questionsfile=Path.home() / "code/haystack/dev/questions.txt",
        top_k=userinput.top_k,
    )

    apikeystring = "OPENAI_API_KEY"
    API_KEY = os.environ.get(apikeystring, None)
    assert API_KEY is not None

    settings = Settings(
        by="word",
        length=userinput.length,
        docstorepath=Path.home() / ".cache/chatlocal",
        retrievertag=userinput.retrievertag,
        embedding_dim=userinput.embedding_dim,
        max_seq_length=userinput.max_seq_length,
        api_key_string=apikeystring,
    )

    # get documentstore
    builder = DocumentstoreBuilder(settings=settings)
    if new_database:
        logger.info(f"creating new docstore with tag {job.tag}")
        builder(job=job)
    if update_db:
        logger.info(f"adding files from {job.datadir} to the docstore")
        builder.add_files(job=job)

    docstore = builder.get_docstore(job=job)
    # get embeddings
    emb = get_embeddings(docstore)
    logger.info(f"got embeddings with shape {emb.v.shape}")

    # extract clusters
    extractorsettings = ExtractorSettings(
        K=userinput.K, blocks=userinput.blocks, spectral_extradims=5
    )
    extractor = ExtractClusters(emb, extractorsettings)
    clusters = extractor(clusterfile=outputdir / "clusters.pkl")

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
    keywordsfile = outputdir / "keywords.txt"

    keywords = kwextractor(clusters, builder, keywordsfile)
    logger.info(f"extracted {len(keywords)} keywords to {keywordsfile}")

    keywordsfile_clean = outputdir / "keywords_cleaned.txt"
    kw_clean = keywordcleanup(
        keywords=keywords, API_KEY=API_KEY, outputfile=keywordsfile_clean
    )
    logger.info(f"extracted {len(kw_clean)} keywords to {keywordsfile_clean}")

    textweightsfile = outputdir / "textweights.pkl"
    if not textweightsfile.exists():
        Wt = sparse_weightmatrix(K=userinput.textgraph_K, emb=emb, metrics=True)
        logger.info(f"saving textweights to {textweightsfile}")
        save_weights(Wt, outputdir / "textweights.pkl")
    else:
        logger.info(f"found {textweightsfile}, skipping textweights creation")
        with textweightsfile.open("rb") as f:
            Wt = pickle.load(f)

    embeddingsfile = outputdir / "kw2textassociation.npy"
    kw_weightsfile = outputdir / "kw_weights_n1.pkl"

    kwgraph = Build_kw_weights(
        keywordsfile=keywordsfile_clean,
        keyword_embeddingfile=embeddingsfile,
        kw_weightsfile=kw_weightsfile,
        apikey=apikeystring,
        n1=userinput.keyword_n,
        n2=userinput.keyword_n,
    )
    kw_weights = kwgraph(emb, Wt)
    logger.info(f"Created kw_weights with {kw_weights.keys()}")


if __name__ == "__main__":
    main(new_database=True)
