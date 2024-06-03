import math
import os
import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from openai import OpenAI
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.semi_supervised import LabelPropagation
from tqdm import tqdm
import networkx as nx

from chatlocal.settings import Embeddings


def save_weights(weights: dict, weightsfile: Path) -> None:
    logger.info(f"Saving weights to {weightsfile}")
    with weightsfile.open("wb") as f:
        pickle.dump(weights, f)


def sparse_weightmatrix(K: int, emb: Embeddings, metrics: bool) -> csr_matrix:
    logger.info(f"Creating sparse weight matrix for texts with K={K}")
    # find the K nearest neighbors for every embedding
    logger.info("Fit KNN model")
    knn = NearestNeighbors(n_neighbors=K + 1, metric="cosine", algorithm="brute")
    knn.fit(emb.v)
    distances, indices = knn.kneighbors(emb.v)
    indices = indices[
        :, 1:
    ]  # remove the first nearest neighbor, which is the text itself

    # Calculate angles and normalization constants
    angles = np.arccos(
        1 - distances[:, 1:]
    )  # arccos to convert cosine distances to angles
    t_values = np.sqrt(
        angles[:, K - 1]
    )  # τ_i = square root of angles to the Kth nearest neighbor

    # Prepare data for sparse matrix
    row_indices = np.repeat(np.arange(emb.v.shape[0]), K)
    col_indices = indices.flatten()
    data = []

    # Calculate the similarity measure
    # for every embedding of text i
    logger.info(f"Calculating similarity measure for {K} nearest texts")
    for i in range(emb.v.shape[0]):
        # for every j up till the K nearest neighbor
        for j in range(K):
            # find the angle <(v_i, v_j)
            angle_ij = angles[i, j]
            # find the normalization constants τ_i and τ_j
            t_i = t_values[i]
            t_j = t_values[indices[i, j]]
            # math.exp is about 20% faster than np.exp
            # calculate the similarity value between textblocks i and j
            similarity = math.exp(-(angle_ij**2) / (t_i * t_j))
            data.append(similarity)

    # Create the sparse matrix W^t
    sparse_matrix = csr_matrix(
        (data, (row_indices, col_indices)), shape=(emb.v.shape[0], emb.v.shape[0])
    )

    ## Make symmetric
    # Step 1: Transpose the matrix
    sparse_matrix_transpose = sparse_matrix.transpose()
    # Step 2: Average the matrix with its transpose
    symmetric_sparse_matrix = (sparse_matrix + sparse_matrix_transpose) / 2
    if metrics:
        calculate_metrics(symmetric_sparse_matrix, K)
    return symmetric_sparse_matrix


def calculate_metrics(W: csr_matrix, K: int) -> None:
    G = nx.from_scipy_sparse_array(W)
    degrees = [d for n, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees)
    diameter = nx.diameter(G)
    logger.info(f"Average degree of the text graph with K={K} : {avg_degree:.2f}")
    logger.info(f"Diameter of the graph : {diameter}")


class Build_kw_weights:
    def __init__(
        self,
        keywordsfile: Path,
        keyword_embeddingfile: Path,
        kw_weightsfile: Path,
        apikey: str,
        n1: int,
        n2: int,
    ):
        self.apikey = apikey
        self.keywords = self.get_keyword(keywordsfile)
        self.keyword_embeddingsfile = keyword_embeddingfile
        self.kw_weightsfile = kw_weightsfile
        self.model = "text-embedding-ada-002"
        self.batch_size = 64
        self.n1 = n1
        self.n2 = n2

    def __call__(self, textembeddings: Embeddings, Wt: csr_matrix) -> dict:
        logger.debug(f"texembeddings type: {type(textembeddings)}")
        # assert isinstance(textembeddings, Embeddings), "textembeddings must be an instance of Embeddings"
        kwemb = self.get_keyword_embeddings(self.keywords)
        closest_indices, farthest_indices = self.find_vectors(kwemb, textembeddings)
        associations = {
            "closest_indices": closest_indices,
            "farthest_indices": farthest_indices,
            "keywords": self.keywords,
        }
        kw_weights = self.build_keyword_weights(associations, textembeddings, Wt)
        return kw_weights

    def get_keyword(self, keywordsfile: Path) -> list[str]:
        with keywordsfile.open("r") as f:
            keywords = f.read().splitlines()
        assert isinstance(keywords, list), "Keywords must be a list"
        return keywords

    def get_keyword_embeddings(self, keywords: list[str]) -> np.ndarray:
        if not self.keyword_embeddingsfile.exists():
            API_KEY = os.environ.get(self.apikey, None)
            client = OpenAI(
                api_key=API_KEY,
            )
            embeddings = []
            for i in tqdm(range(0, len(keywords), self.batch_size)):
                batch = keywords[i : i + self.batch_size]
                response = client.embeddings.create(input=batch, model=self.model)
                embeddings.extend([item.embedding for item in response.data])
            key_embeddings = np.stack(embeddings)
            logger.info(f"Saving embeddings to {self.keyword_embeddingsfile}")
            np.save(self.keyword_embeddingsfile, key_embeddings)
        else:
            logger.info(f"Loading embeddings from {self.keyword_embeddingsfile}")
            key_embeddings = np.load(self.keyword_embeddingsfile)
        return key_embeddings

    def find_vectors(
        self, keyword_embeddings: np.ndarray, emb: Embeddings
    ) -> tuple[np.ndarray, np.ndarray]:
        logger.info(
            f"calculating similarity for {keyword_embeddings.shape[0]}x{emb.v.shape[0]} keywords x texts"
        )
        # Calculate cosine similarity (1 - cosine distance)
        similarity = 1 - cdist(keyword_embeddings, emb.v, "cosine")
        # embeddings shape: (keywords, 1536), emb.v shape: (texts, 1536)
        # similarity shape: (keywords, texts)

        # Initialize arrays to store the indices of closest and farthest vectors
        logger.info(
            f"initializing arrays for {self.n1} closest and {self.n2} farthest indices"
        )
        closest_indices = np.zeros((keyword_embeddings.shape[0], self.n1), dtype=int)
        farthest_indices = np.zeros((keyword_embeddings.shape[0], self.n2), dtype=int)

        # For each keyword, find the n1 closest and n2 farthest vectors
        logger.info(
            f"assigning {self.n1} closest and {self.n2} farthest indices for every keyword"
        )
        for i in tqdm(range(similarity.shape[0])):
            # tried to vectorize np.argsort, but it gives unexpected results
            sorted_indices = np.argsort(-similarity[i])  # Descending order
            closest_indices[i] = sorted_indices[: self.n1]
            farthest_indices[i] = sorted_indices[-self.n2 :]

        # Labeling: 1 for closest, 0 for farthest, -1 for the rest
        # logger.info("Assigning labels (1 for closest, 0 for farthest, -1 for the rest)")
        # labels = np.zeros(emb_v.shape[0], dtype=int) - 1
        # for i, indices in enumerate(closest_indices):
        #     labels[indices] = 1
        # for i, indices in enumerate(farthest_indices):
        #     labels[indices] = 0

        return closest_indices, farthest_indices

    def build_keyword_weights(
        self, associations: dict, emb: Embeddings, Wt: csr_matrix
    ) -> dict:
        if not self.kw_weightsfile.exists():
            logger.info(
                f"{self.kw_weightsfile} does not exist, building keyword weights"
            )
            closest_indices = associations["closest_indices"]
            farthest_indices = associations["farthest_indices"]
            keywords = associations["keywords"]

            M = len(keywords)
            N = emb.v.shape[0]

            labels = np.zeros((M, N), dtype=int) - 1
            predictions = []
            logger.info(f"Doing label propagation for {M} keywords")
            for i in tqdm(range(M)):
                labels[i, closest_indices[i]] = 1
                labels[i, farthest_indices[i]] = 0
                gaussian = LabelPropagation()
                gaussian.fit(Wt, labels[i])
                yhat = gaussian.transduction_
                predictions.append(yhat)
            propagated_labels = np.stack(predictions)
            pl_sparse = csr_matrix(propagated_labels)
            Wk = pl_sparse.dot(pl_sparse.T)
            Wk.setdiag(0)
            kw_weights = {"Wk": Wk, "propagated_labels": propagated_labels}
            self.save_weights(kw_weights)
        else:
            logger.info(f"Loading keyword weights from {self.kw_weightsfile}")
            with self.kw_weightsfile.open("rb") as f:
                kw_weights = pickle.load(f)
        logger.info(f"Obtained kw_weights: {kw_weights.keys()}")
        logger.info(f"propagated_labels shape: {kw_weights['propagated_labels'].shape}")
        logger.info(f"Wk shape: {kw_weights['Wk'].shape}")

        return kw_weights

    def save_weights(self, kw_weights: dict):
        logger.info(f"Saving keyword weights to {self.kw_weightsfile}")
        with self.kw_weightsfile.open("wb") as f:
            pickle.dump(kw_weights, f)
