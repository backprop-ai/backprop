from typing import Callable, Dict, List
from sentence_transformers import SentenceTransformer

from .documents import DocStore, Document, ElasticDocStore
from .utils import vectorize_document


class Kiri:
    """Core class of natural language engine"""

    def __init__(self, store: DocStore, vectorize_model: str = None,
                 vectorize_func: Callable[[Document, SentenceTransformer], List[float]] = None):
        """Initializes internal state"""

        if store is None:
            raise ValueError("a DocStore implementation must be provided")

        if vectorize_model is None:
            # Use default vectorization model
            vectorize_model = "distiluse-base-multilingual-cased-v2"

        if vectorize_func is None:
            # Use default vectorizer
            vectorize_func = vectorize_document

        self._store = store
        self._vectorize_func = vectorize_func
        self._vectorize_model = SentenceTransformer(vectorize_model)

    def upload(self, documents: List[Document]) -> None:
        """Upload documents to store"""

        return self._store.upload(documents, self._vectorize_func,
                                  self._vectorize_model)

    def search(self, query: str, ids=None, body=None):
        """
        Search documents from document store
        """
        return self._store.search(query, self._vectorize_model, ids=ids, body=body)
