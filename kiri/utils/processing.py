from sentence_transformers import SentenceTransformer
from nltk import sent_tokenize
from scipy.spatial.distance import cdist
from typing import List, Tuple

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


def calc_doc_score(max_score: float, doc_score: float, chunk_scores: Tuple[str, float], top_chunks: int = 3):
    doc_scores = []

    norm_doc_score = doc_score / max_score

    for chunk, score in chunk_scores[:top_chunks]:
        doc_scores.append(score)

    avg_chunk_score = sum(doc_scores) / len(doc_scores)

    final_score = (norm_doc_score * avg_chunk_score + avg_chunk_score) / 2
    return final_score


def calc_chunk_scores(chunks: List[str], chunk_vectors: List[List[float]], query_vec: List[float]):
    distances = cdist(
        [query_vec], chunk_vectors, "cosine")[0]
    scores = [1.0 - d for d in distances]
    chunk_scores = zip(chunks, scores)

    # Highest score first
    chunk_scores = sorted(chunk_scores, key=lambda x: x[1], reverse=True)

    return chunk_scores


def gen_preview_from_chunks(chunks: List[str], preview_length: int):
    preview = ""
    reached_length = False
    for chunk in chunks:
        words = chunk.split()
        for word in words:
            if len(preview) + len(word) <= preview_length:
                preview += f" {word}"
            else:
                reached_length = True
                break

        if reached_length:
            break

    return preview


def process_results(search_results, query_vec, doc_class, preview_length: int):
    """
    Process search results based on document type
    """
    max_score = search_results.max_score

    for result in search_results.results:
        document = result.document
        # Only do processing for known document classes
        if issubclass(doc_class, ChunkedDocument):
            chunk_scores = calc_chunk_scores(document.chunks, document.chunk_vectors,
                                             query_vec)
            score = calc_doc_score(
                search_results.max_score, result.score, chunk_scores)
            result.score = score
            if score > max_score:
                max_score = score

            # Chunks ordered by highest score
            top_chunks = [cs[0] for cs in chunk_scores]
            preview = gen_preview_from_chunks(top_chunks, preview_length)
            result.preview = preview
        elif issubclass(doc_class, Document):
            # Treat entire content as a chunk
            preview = gen_preview_from_chunks(
                [document.content], preview_length)
            result.preview = preview

    search_results.results = sorted(
        search_results.results, key=lambda r: r.score, reverse=True)

    search_results.max_score = max_score
