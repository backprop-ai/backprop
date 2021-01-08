from kiri import Kiri, Document, ChunkedDocument
from kiri.search import SearchResult
import pytest


def get_docs():
    doc1 = Document("Hello I am a document. This is a sentence")
    doc2 = Document("Hello I am another document. This is another sentence")
    docs = [doc1, doc2]
    return docs


def get_chunked_docs(chunking_level=1):
    doc1 = ChunkedDocument(
        "Hello I am a document. This is a sentence", chunking_level=chunking_level)
    doc2 = ChunkedDocument(
        "Hello I am another document. This is another sentence", chunking_level=chunking_level)
    docs = [doc1, doc2]
    return docs


def test_init():
    kiri = Kiri(local=True)


def test_upload():
    kiri = Kiri(local=True)
    docs = get_docs()
    kiri.upload(docs)
    assert docs[0].vector is not None, "Document not vectorised"
    assert docs[1].vector is not None, "Document not vectorised"
    assert len(kiri._store.documents) == 2, "Incorrect number of documents in mem"


def test_upload_chunked():
    kiri = Kiri(local=True)
    docs = get_chunked_docs(chunking_level=1)
    kiri.upload(docs)
    assert len(kiri._store.documents) == 2, "Incorrect number of documents in mem"
    for doc in docs:
        assert doc.vector is not None, "Document not vectorised"
        assert len(doc.chunk_vectors) == 2, "Invalid number of chunk vectors"


def test_upload_dup_id():
    kiri = Kiri(local=True)
    docs = get_docs()
    for doc in docs:
        doc.id = "123"

    with pytest.raises(ValueError):
        kiri.upload(docs)


def test_upload_mixed_type():
    kiri = Kiri(local=True)
    docs = [Document("a"), ChunkedDocument("b")]

    with pytest.raises(ValueError):
        kiri.upload(docs)


def test_search():
    kiri = Kiri(local=True)
    docs = get_docs()
    kiri.upload(docs)
    results = kiri.search("another")
    assert len(results.results) == 2, "Invalid number of search results"


def test_search_max_results():
    kiri = Kiri(local=True)
    docs = get_docs()
    kiri.upload(docs)
    results = kiri.search("another", max_results=1)
    assert len(results.results) == 1, "Invalid number of search results"


def test_search_ids():
    kiri = Kiri(local=True)
    docs = get_docs()
    docs[0].id = "123"
    kiri.upload(docs)
    results = kiri.search("another", ids=["123"])
    assert len(results.results) == 1, "Invalid number of search results"


def test_search_chunk():
    kiri = Kiri(local=True)
    docs = get_chunked_docs()
    kiri.upload(docs)
    results = kiri.search("another")
    assert len(results.results) == 2, "Invalid number of search results"


def test_search_max_results_chunk():
    kiri = Kiri(local=True)
    docs = get_chunked_docs()
    kiri.upload(docs)
    results = kiri.search("another", max_results=1)
    assert len(results.results) == 1, "Invalid number of search results"


def test_search_ids_chunk():
    kiri = Kiri(local=True)
    docs = get_chunked_docs()
    docs[0].id = "123"
    kiri.upload(docs)
    results = kiri.search("another", ids=["123"])
    assert len(results.results) == 1, "Invalid number of search results"


def test_qa():
    kiri = Kiri(local=True)
    docs = get_docs()
    kiri.upload(docs)
    results = kiri.qa("another?")
    assert isinstance(results, list)
    for result in results:
        assert type(result[0]) == str
        assert isinstance(result[1], SearchResult)


def test_qa_chunk():
    kiri = Kiri(local=True)
    docs = get_chunked_docs()
    kiri.upload(docs)
    results = kiri.qa("another?")
    assert isinstance(results, list)
    for result in results:
        assert type(result[0]) == str
        assert isinstance(result[1], SearchResult)
