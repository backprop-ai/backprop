from sentence_transformers import SentenceTransformer

from ..documents import Document


def vectorize_document(document: Document, model: SentenceTransformer):
    """
    Vectorize document based on type
    """
    if type(document) == Document:
        return model.encode(document.content)
    else:
        raise ValueError(
            f"vectorisation of document of type {type(document)} is not implemented")
