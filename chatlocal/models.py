from haystack.document_stores.base import BaseDocumentStore
from settings import Embeddings
import numpy as np
from loguru import logger


def get_embeddings(docstore: BaseDocumentStore) -> Embeddings:
    logger.info("Retrieving embeddings")
    docs = docstore.get_all_documents(return_embedding=True)
    e = np.stack([d.embedding for d in docs])
    logger.info(f"Text embeddings shape: {e.shape}")
    ids = [d.id for d in docs]
    return Embeddings(e, ids)
