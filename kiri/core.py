from typing import Callable, Dict, List
from sentence_transformers import SentenceTransformer

from .search import DocStore, SearchResults, Document
from .utils import process_document, process_results


class Kiri:
    """Core class of natural language engine"""

    def __init__(self, store: DocStore, vectorize_model: str = None,
                 process_doc_func: Callable[[
                     Document, SentenceTransformer], List[float]] = None,
                 process_results_func: Callable[[SearchResults, SentenceTransformer], None] = None):
        """Initializes internal state"""

        if store is None:
            raise ValueError("a DocStore implementation must be provided")

        if vectorize_model is None:
            # Use default vectorization model
            vectorize_model = "ojasaar/distilbert-sentence-msmarco-en-et"

        if process_doc_func is None:
            # Use default vectorizer
            process_doc_func = process_document

        if process_results_func is None:
            process_results_func = process_results

        self._store = store
        self._process_doc_func = process_doc_func
        self._process_results_func = process_results
        self._vectorize_model = SentenceTransformer(vectorize_model)

    def upload(self, documents: List[Document]) -> None:
        """Upload documents to store"""

        return self._store.upload(documents, self._process_doc_func,
                                  self._vectorize_model)

    def search(self, query: str, max_results=10, min_score=0.0,
               preview_length=100, ids=None, body=None) -> SearchResults:
        """
        Search documents from document store
        """
        search_results, query_vec = self._store.search(query, self._vectorize_model,
                                                       max_results=max_results, min_score=min_score,
                                                       ids=ids, body=body)
        self._process_results_func(
            search_results, query_vec, self._store._doc_class, preview_length)
        return search_results
