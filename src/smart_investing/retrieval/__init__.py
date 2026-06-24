"""Thematic retrieval: hybrid (dense+sparse) search + supply-chain graph -> universe."""

from smart_investing.retrieval.embeddings import TfidfEmbedder, get_embedder
from smart_investing.retrieval.graph import RelationGraph, build_comention_graph
from smart_investing.retrieval.index import RetrievalIndex
from smart_investing.retrieval.universe import build_index_from_store, build_universe

__all__ = [
    "get_embedder",
    "TfidfEmbedder",
    "RetrievalIndex",
    "RelationGraph",
    "build_comention_graph",
    "build_universe",
    "build_index_from_store",
]
