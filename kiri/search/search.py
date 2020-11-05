from elasticsearch import Elasticsearch, helpers
from typing import Dict, List
import shortuuid
from ..utils import elastic_to_search_results
from .documents import Document, ElasticDocument

import time


class SearchResult:
    def __init__(self, document: Document, score: float, preview: str = None):
        self.document = document
        self.score = score
        self.preview = preview

    def to_json(self, exclude_vectors=True):
        # TODO: implement excluding vectors
        json_repr = vars(self)
        json_repr["document"] = vars(json_repr["document"])
        return json_repr


class SearchResults:
    def __init__(self, max_score: float, total_results: int, results: List[SearchResult]):
        self.max_score = max_score
        self.total_results = total_results
        self.results = results

    def to_json(self, exclude_vectors=True):
        # TODO: implement excluding vectors
        json_repr = vars(self)
        json_repr["results"] = [
            r.to_json(exclude_vectors=exclude_vectors) for r in json_repr["results"]]
        return json_repr


class DocStore:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("__init__ is not implemented!")

    def upload(self, *args, **kwargs):
        raise NotImplementedError("upload is not implemented!")

    def search(self, *args, **kwargs):
        raise NotImplementedError("search is not implemented!")


class ElasticDocStore(DocStore):
    def __init__(self, url: str, index: str = "kiri_default",
                 doc_class: ElasticDocument = ElasticDocument):
        self._client = Elasticsearch([url])
        self._index = index
        self._doc_class = doc_class

        correct_mapping = doc_class.elastic_mappings()

        # Setup index with correct mappings
        if self._client.indices.exists(self._index):
            # Check if mapping is correct
            mapping = self._client.indices.get_mapping(
                self._index).get(self._index).get("mappings")

            # Try updating mapping
            if mapping != correct_mapping:
                self._client.indices.close(self._index)
                self._client.indices.put_mapping(
                    correct_mapping, index=self._index)
                self._client.indices.open(self._index)

        else:
            # Create index with mapping
            self._client.indices.create(
                self._index, body={"mappings": correct_mapping})

    def upload(self, documents: List[ElasticDocument], vectorize_func, vectorize_model, index: str = None) -> None:
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
                vectorize_func(document, vectorize_model)

            # JSON representation of document
            doc_json = document.to_elastic()

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

    def search(self, query, vectorize_model, max_results=10, min_score=0.0, ids=None, body=None):
        """
        Search documents from elasticsearch
        """
        query_vec = vectorize_model.encode(query)

        # elasticsearch does not support negative scores
        score_modifier = 1.0

        if body is None:
            body = {
                "min_score": min_score + score_modifier,
                "size": max_results,
                "query": {
                    "function_score": {
                        "query": {
                            "bool": {
                                "should": [
                                    {"match": {
                                        "content": query
                                    }},
                                    {"match_all": {}}

                                ]
                            }

                        },
                        "script_score": {
                            "script": {
                                "source": f"(cosineSimilarity(params.query_vector, 'vector') + {score_modifier}) * (_score + 1)",
                                "params": {
                                    "query_vector": query_vec.tolist()}
                            }
                        },
                    }
                },
                "highlight": {
                    "pre_tags": ["<b>"],
                    "post_tags": ["</b>"],
                    "fragment_size": 100,
                    "fields": {
                        "content": {}
                    }
                },

            }

        start = time.time()
        res = self._client.search(index=self._index, body=body)
        search_results = elastic_to_search_results(
            res, score_modifier, self._doc_class)

        return search_results, query_vec
