from typing import Callable


class Document:
    pass


class DocStore:
    pass


def vectorize(document: Document) -> str:

    # TODO: vectorise given document

    return ""


class Kiri:
    """Core class of natural language engine"""

    def __init__(self, store: DocStore, vectorize_func: Callable[[[Document]], str]):
        """Initializes internal state"""

        if store is None:
            raise ValueError("a DocStore implementation must be provided")

        if vectorize_func is None:
            raise ValueError("a Vectorizer implementation must be provided")

        self._store = store
        self._vectorize_func = vectorize_func

    def upload(self, documents: [Document]) -> None:
        """Upload documents to store"""

        self._store.upload(documents)
