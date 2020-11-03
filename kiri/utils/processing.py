from sentence_transformers import SentenceTransformer
from nltk import sent_tokenize

from ..search import Document, ChunkedDocument


def get_sentences(content):
    sentences = sent_tokenize(content)

    return sentences


def chunk_document(document: ChunkedDocument):
    """
    Return list of document chunks based on chunking level
    """

    chunking_level = document.chunking_level

    # Split to sentences
    sentences = get_sentences(document.content)

    # Join groups of sentences into chunks by chunking level
    # ["a", "b", "c"] with chunking level 2
    # creates ["a b", "c"]
    chunks = [" ".join(sentences[i:i + chunking_level])
              for i in range(0, len(sentences), chunking_level)]

    return chunks


def process_document(document: Document, model: SentenceTransformer):
    """
    Process document based on type
    """
    if isinstance(document, ChunkedDocument):
        chunks = chunk_document(document)
        chunk_vectors = model.encode(chunks)
        document.chunks = chunks
        document.chunk_vectors = chunk_vectors
        document.vector = model.encode(document.content)
    elif isinstance(document, Document):
        document.vector = model.encode(document.content)
    else:
        raise ValueError(
            f"vectorisation of document of type {type(document)} is not implemented")
