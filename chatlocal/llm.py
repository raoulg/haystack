from builder import DocumentstoreBuilder
from settings import Clusters


def text_from_cluster(cluster: Clusters, builder: DocumentstoreBuilder) -> dict:
    d = {}
    for i in range(len(cluster.center)):
        d[i] = builder.document_store.get_documents_by_id(ids=cluster.center[i])
    for i in range(len(cluster.random)):
        d[i] += builder.document_store.get_documents_by_id(ids=cluster.random[i])
    return d
