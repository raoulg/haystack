import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from settings import Clusters, Embeddings, ExtractorSettings
from sklearn.cluster import KMeans
from sklearn.manifold import spectral_embedding
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import KDTree

from haystack.document_stores.base import BaseDocumentStore


def get_embeddings(docstore: BaseDocumentStore) -> Embeddings:
    logger.info("Retrieving embeddings")
    docs = docstore.get_all_documents(return_embedding=True)
    e = np.stack([d.embedding for d in docs])
    logger.info(f"Text embeddings shape: {e.shape}")
    ids = [d.id for d in docs]
    return Embeddings(v=e, ids=ids)


class ExtractClusters:
    def __init__(self, emb: Embeddings, settings: ExtractorSettings):
        self.emb = emb
        self.K = settings.K
        self.blocks = settings.blocks
        self.extra_dims = settings.spectral_extradims
        self.c1 = None
        self.c2 = None

    def __call__(self, clusterfile: Path) -> dict:
        """Returns a dictionary with keys 'kmeans' and 'spectral',
        For every clustering algorithm, there are two groups selected:
            1. The K items, closest to the cluster center
            2. K random items from the cluster
        A list with K ids are stored in a dictionary, with the cluster label as key.
        The dictionaries are, in turn, stored in a Clusters dataclass


        containing indices of the K neighbors for every cluster center from KMeans and Spectral Clustering
        """
        if not clusterfile.exists():
            clusters = {"kmeans": self.kmeans(), "spectral": self.spectral()}
            self.save_clusters(clusters, clusterfile)
        else:
            logger.info(f"Loading clusters from {clusterfile}")
            with clusterfile.open("rb") as f:
                clusters = pickle.load(f)
        return clusters

    def kdsearch(self, emb: np.ndarray, centers) -> dict:
        tree = KDTree(emb)
        _, ind = tree.query(centers, k=self.blocks)
        return {label: [self.emb.ids[i] for i in ind[label]] for label in range(self.K)}

    def random_samples(self, labels: np.ndarray) -> dict:
        unique_labels = list(range(self.K))
        random_samples = {}
        for label in unique_labels:
            cluster_samples = np.random.choice(
                np.where(labels == label)[0], self.blocks, replace=False
            )
            random_samples[label] = [self.emb.ids[i] for i in cluster_samples]
        return random_samples

    def kmeans(self) -> Clusters:
        logger.info("Starting KMeans")
        self.c1 = KMeans(n_clusters=self.K, init="k-means++", n_init=5)
        assert self.c1 is not None
        self.c1.fit(self.emb.v)
        logger.info("Searching KDtree for KMeans")
        centers = self.kdsearch(self.emb.v, self.c1.cluster_centers_)
        random = self.random_samples(self.c1.labels_)
        clusters = Clusters(center=centers, random=random)
        return clusters

    def spectral(self) -> Clusters:
        logger.info("Starting Spectral Clustering")
        kernel = rbf_kernel(self.emb.v, gamma=1.0)
        maps = spectral_embedding(
            kernel, n_components=self.K + self.extra_dims, eigen_solver="arpack"
        )
        km = KMeans(n_clusters=self.K, n_init="auto")

        self.c2 = km.fit(maps)
        assert self.c2 is not None

        logger.info("Searching KDtree for Spectral Clustering")
        centers = self.kdsearch(maps, self.c2.cluster_centers_)
        random = self.random_samples(self.c2.labels_)
        clusters = Clusters(center=centers, random=random)
        return clusters

    def save_clusters(self, clusters: dict, clusterfile: Path) -> None:
        logger.info(f"Saving clusters to {clusterfile}")
        with clusterfile.open("wb") as f:
            pickle.dump(clusters, f)
