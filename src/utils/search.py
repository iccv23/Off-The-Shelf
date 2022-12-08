import time
import faiss
from faiss.contrib import inspect_tools
import numpy as np
import torch


def assignCodebook(pq: faiss.IndexPQ, codebook):
    originalCodebook = inspect_tools.get_pq_centroids(pq.pq)
    if originalCodebook.shape != codebook.shape:
        raise ValueError(f"Codebook shape mismatch, expected {originalCodebook.shape}, but given {codebook.shape}.")
    faiss.copy_array_to_vector(codebook, pq.pq.centroids)
    if ((codebook - inspect_tools.get_pq_centroids(pq.pq)) ** 2).sum() > 1e-10:
        raise ValueError("Assign failed.")

@torch.no_grad()
def rank_list_pq(database: np.ndarray, query: np.ndarray, codebook: np.ndarray, R: int) -> np.ndarray:
    m, k, d = codebook.shape
    pq = faiss.IndexPQ(m * d, m, 8)

    pq.train(query)

    # assignCodebook(pq, codebook)

    pq.add(database)

    start_time = time.time()
    indices = pq.assign(query, R, None)
    end_time = time.time()

    print(f"Search in {len(database)} database, R = {R}: {(end_time - start_time) * 1000 / len(query)} millisecond per query.")

    return indices

@torch.no_grad()
def rank_list_hash(database, query, R: int):
    bits = 64
    database =  np.empty((len(database), bits // 8), dtype="uint8")
    query =  np.empty((len(query), bits // 8), dtype="uint8")
    index = faiss.IndexBinaryFlat(bits)
    index.add(database)
    start_time = time.time()
    D, I = index.search(query, R)
    end_time = time.time()

    print(f"Search in {bits}-bits {len(database)} database, R = {R}: {(end_time - start_time) * 1000 / len(query)} millisecond per query.")

    return I

@torch.no_grad()
def rank_list_brute_force(database: np.ndarray, query: np.ndarray, R: int) -> np.ndarray:
    flat = faiss.IndexFlatL2(database.shape[-1])

    flat.add(database)

    start_time = time.time()
    indices = flat.assign(query, R, None)
    end_time = time.time()

    print(f"Search in {len(database)} database, R = {R}: {(end_time - start_time) * 1000 / len(query)} millisecond per query.")

    return indices
