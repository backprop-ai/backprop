from kiri import Kiri, ElasticDocument, ElasticChunkedDocument, ElasticDocStore
from kiri.search import SearchResult
import pytest
import requests


def get_docs():
    doc1 = ElasticDocument("Hello I am a document. This is a sentence")
    doc2 = ElasticDocument(
        "Hello I am another document. This is another sentence")
    docs = [doc1, doc2]
    return docs


def get_chunked_docs(chunking_level=1):
    doc1 = ElasticChunkedDocument(
        "Hello I am a document. This is a sentence", chunking_level=chunking_level)
    doc2 = ElasticChunkedDocument(
        "Hello I am another document. This is another sentence", chunking_level=chunking_level)
    docs = [doc1, doc2]
    return docs


def init_elastic_kiri(doc_class=ElasticDocument):
    index_name = "temp_test"
    elastic_url = "http://localhost:9200"
    requests.delete(f"{elastic_url}/{index_name}")
    store = ElasticDocStore(elastic_url, index=index_name, doc_class=doc_class)
    kiri = Kiri(store=store, local=True)
    return kiri


def test_init():
    kiri = init_elastic_kiri()


def test_init_chunked():
    kiri = init_elastic_kiri(doc_class=ElasticChunkedDocument)


def test_upload():
    kiri = init_elastic_kiri()
    docs = get_docs()
    kiri.upload(docs)
    for doc in docs:
        assert doc.vector is not None, "Document not vectorised"
        assert doc.vector is not None, "Document not vectorised"


def test_upload_chunked():
    kiri = init_elastic_kiri()
    docs = get_chunked_docs(chunking_level=1)
    kiri.upload(docs)
    for doc in docs:
        assert doc.vector is not None, "Document not vectorised"
        assert len(doc.chunk_vectors) == 2, "Invalid number of chunk vectors"


def test_upload_dup_id():
    kiri = init_elastic_kiri()
    docs = get_docs()
    for doc in docs:
        doc.id = "123"

    with pytest.raises(ValueError):
        kiri.upload(docs)


def test_upload_mixed_type():
    kiri = init_elastic_kiri()
    docs = [ElasticDocument("a"), ElasticChunkedDocument("b")]

    with pytest.raises(ValueError):
        kiri.upload(docs)


def test_search():
    kiri = init_elastic_kiri()
    docs = get_docs()
    kiri.upload(docs)
    results = kiri.search("another")
    assert len(results.results) == 2, "Invalid number of search results"


def test_search_max_results():
    kiri = init_elastic_kiri()
    docs = get_docs()
    kiri.upload(docs)
    results = kiri.search("another", max_results=1)
    assert len(results.results) == 1, "Invalid number of search results"


def test_search_ids():
    kiri = init_elastic_kiri()
    docs = get_docs()
    docs[0].id = "123"
    kiri.upload(docs)
    results = kiri.search("another", ids=["123"])
    assert len(results.results) == 1, "Invalid number of search results"


def test_search_chunk():
    kiri = init_elastic_kiri(doc_class=ElasticChunkedDocument)
    docs = get_chunked_docs()
    kiri.upload(docs)
    results = kiri.search("another")
    assert len(results.results) == 2, "Invalid number of search results"


def test_search_max_results_chunk():
    kiri = init_elastic_kiri(doc_class=ElasticChunkedDocument)
    docs = get_chunked_docs()
    kiri.upload(docs)
    results = kiri.search("another", max_results=1)
    assert len(results.results) == 1, "Invalid number of search results"


def test_search_ids_chunk():
    kiri = init_elastic_kiri(doc_class=ElasticChunkedDocument)
    docs = get_chunked_docs()
    docs[0].id = "123"
    kiri.upload(docs)
    results = kiri.search("another", ids=["123"])
    assert len(results.results) == 1, "Invalid number of search results"


def test_qa():
    kiri = init_elastic_kiri()
    docs = get_docs()
    kiri.upload(docs)
    results = kiri.qa("another?")
    assert isinstance(results, list)
    for result in results:
        assert type(result[0]) == str
        assert isinstance(result[1], SearchResult)


def test_qa_chunk():
    kiri = init_elastic_kiri(doc_class=ElasticChunkedDocument)
    docs = get_chunked_docs()
    kiri.upload(docs)
    results = kiri.qa("another?")
    assert isinstance(results, list)
    for result in results:
        assert type(result[0]) == str
        assert isinstance(result[1], SearchResult)
