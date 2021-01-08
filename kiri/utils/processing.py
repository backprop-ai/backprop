from sentence_transformers import SentenceTransformer
from nltk import sent_tokenize
from scipy.spatial.distance import cdist
from typing import List, Tuple, Union

from ..search import Document, ChunkedDocument


def batch_items(items, batch_size=16):
    batches = [items[i:i + batch_size]
               for i in range(0, len(items), batch_size)]
    return batches


def get_sentences(content):
    """Tokenizes document content into sentences

    Args:
        content: String of document contents

    Returns:
        list of strings, the separate sentences of <content>
    """
    sentences = sent_tokenize(content)

    return sentences


def chunk_document(document: ChunkedDocument):
    """Gets document chunks based on chunking level

    Args:
        document: Document to be chunked 

    Returns:
        Chunks to be assigned to <document>
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


def process_documents(documents: Union[Document, List[Document]], doc_store):
    """Processes document based on type

    Args:
        documents: Document or list of Documents to be processed
        model: SentenceTransformer model name used for processing
        doc_store: Document store that holds the documents

    Raises:
        ValueError: If the given document's type doesn't have a vectorisation function
    """
    # Example for instance verification
    doc_example = documents[0]
    if isinstance(doc_example, Document):
        contents = [d.content for d in documents]
        content_vectors = doc_store.kiri.vectorise(contents)

        # Set vectors for each document
        for document, vector in zip(documents, content_vectors):
            document.vector = vector

    # TODO: Batching for document chunks
    if isinstance(doc_example, ChunkedDocument):
        for document in documents:
            chunks = chunk_document(document)
            chunk_vectors = doc_store.kiri.vectorise(chunks)
            document.chunks = chunks
            document.chunk_vectors = chunk_vectors

    if not isinstance(doc_example, Document) and not isinstance(doc_example, ChunkedDocument):
        raise ValueError(
            f"vectorisation of document of type {type(document)} is not implemented")


def calc_doc_score(max_score: float, doc_score: float, chunk_scores: Tuple[str, float], top_chunks: int = 3):
    # def calc_doc_score(doc_score: float, chunk_scores: Tuple[str, float], top_chunks: int = 3):
    """Calculates the overall score of a chunked document

    Args:
        max_score: The maximum score returned by the search, for normalization
        doc_score: The base relevancy score from Elasticsearch
        chunk_scores: Tuple of each chunk with its corresponding score
        top_chunks: number of chunks to be included in the weighting

    Returns:
        The final weighted score for the document
    """

    doc_scores = []

    norm_doc_score = doc_score / max_score

    for chunk, score in chunk_scores[:top_chunks]:
        doc_scores.append(score)

    avg_chunk_score = sum(doc_scores) / len(doc_scores)

    final_score = (norm_doc_score * avg_chunk_score + avg_chunk_score) / 2
    return final_score


def calc_chunk_scores(chunks: List[str], chunk_vectors: List[List[float]], query_vec: List[float]):
    """Calculates score of each chunk of a document

    Args:
        chunks: List of sentence chunks on the desired document
        chunk_vectors: List of vectors for each chunk in the document
        query_vec: vectorised query string

    Returns:
        Sorted list of chunks, with the highest first.
    """

    distances = cdist(
        [query_vec], chunk_vectors, "cosine")[0]
    scores = [1.0 - d for d in distances]
    chunk_scores = zip(chunks, scores)

    # Highest score first
    chunk_scores = sorted(chunk_scores, key=lambda x: x[1], reverse=True)

    return chunk_scores


def gen_preview_from_chunks(chunks: List[str], preview_length: int):
    """Generates a preview string for a document

    Args:
        chunks: List of strings, the chunks of the document
        preview_length: Number of characters in the preview

    Returns:
        Preview/metatext string
    """
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


def process_results(search_results, query_vec, doc_class, preview_length: int,
                    max_results: int, min_score: float):
    """Calculates final scores for returned search results, updates SearchResults

    Args:
        search_results: Kiri SearchResults object.
        query_vec: Vectorised query string
        doc_class: Type of Document being processed
        preview_length: Number of characters in the preview string
    """
    max_score = -1.0
    all_chunks = []

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
            # Keep track of all chunks
            all_chunks += [{"chunk": cs[0], "score": cs[1],
                            "search_result": result} for cs in chunk_scores]
            preview = gen_preview_from_chunks(top_chunks, preview_length)
            result.preview = preview
        elif issubclass(doc_class, Document):
            # Treat entire content as a chunk
            chunk_scores = calc_chunk_scores([document.content], [document.vector],
                                             query_vec)
            score = calc_doc_score(
                search_results.max_score, result.score, chunk_scores)
            result.score = score
            if score > max_score:
                max_score = score

            preview = gen_preview_from_chunks(
                [document.content], preview_length)
            result.preview = preview

    # Order by decreasing score, up to max_results
    search_results.results = sorted(
        search_results.results, key=lambda r: r.score, reverse=True)[:max_results]

    # Order chunks by decreasing score, up to max_results
    search_results.top_chunks = sorted(
        all_chunks, key=lambda c: c["score"], reverse=True)[:max_results]

    # Remove documents with score < min_score
    search_results.results = [
        r for r in search_results.results if r.score >= min_score]

    search_results.max_score = max_score
