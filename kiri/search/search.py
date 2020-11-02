from elasticsearch import Elasticsearch, helpers
from typing import Dict, List
import shortuuid
from ..utils import elastic_to_search_results
from .documents import Document, ElasticDocument


class SearchResult:
    def __init__(self, document: Document, score: float, preview: str = None):
        self.document = document
        self.score = score
        self.preview = preview


class SearchResults:
    def __init__(self, max_score: float, total_results: int, results: List[SearchResult]):
        self.max_score = max_score
        self.total_results = total_results
        self.results = results


class DocStore:
    def __init__(self, params):
        raise NotImplementedError("The constructor is not implemented!")

    def upload(self, params):
        raise NotImplementedError

    def search(self, params):
        raise NotImplementedError


class ElasticDocStore(DocStore):
    def __init__(self, url: str, index: str = "kiri_default",
                 doc_class: ElasticDocument = ElasticDocument):
        self._client = Elasticsearch([url])
        self._index = index

        correct_mapping = doc_class.elastic_mappings()

        # Setup index with correct mappings
        if self._client.indices.exists(self._index):
            # Check if mapping is correct
            mapping = self._client.indices.get_mapping(
                self._index).get(self._index).get("mappings")

            if mapping != correct_mapping:
                # Update mapping
                self._client.indices.close(self._index)
                self._client.indices.put_mapping(
                    correct_mapping, index=self._index)
                self._client.indices.open(self._index)

        else:
            # Create index with mapping
            self._client.indices.create(
                self._index, body={"mappings": correct_mapping})

    def upload(self, documents: List[Document], vectorize_func, vectorize_model, index: str = None) -> None:
        """
        Upload documents to elasticsearch
        """
        if not index:
            index = self._index

        # TODO: Check ID uniqueness

        # TODO: Batching

        payload = []
        for document in documents:
            # Calculate missing vectors
            if document.vector is None:
                document.vector = vectorize_func(document, vectorize_model)

            # JSON representation of document
            doc_json = document.__dict__

            # Add correct index
            doc_json["_index"] = index

            # Rename id key
            doc_json["_id"] = doc_json["id"]
            del doc_json["id"]

            payload.append(doc_json)

        # Bulk upload to elasticsearch
        helpers.bulk(self._client, payload)
        # Update index
        self._client.indices.refresh(index=self._index)

    def search(self, query, vectorize_model, max_results=10, ids=None, body=None):
        """
        Search documents from elasticsearch
        """
        query_vec = vectorize_model.encode(query)

        if body is None:
            body = {
                "size": max_results,
                "query": {
                    "script_score": {
                        "query": {
                            "match_all": {}
                        },

                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                            "params": {
                                "query_vector": query_vec.tolist()}
                        }

                    },
                }
                # "aggs": {
                #     "types_count": {"value_count": {"field": "url"}}
                # }
            }

        res = self._client.search(index=self._index, body=body)
        search_results = elastic_to_search_results(res)

        return search_results
