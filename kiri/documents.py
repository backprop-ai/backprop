from elasticsearch import Elasticsearch, helpers
from typing import Dict, List
import shortuuid


class Document:
    def __init__(self, content: str, id: str = shortuuid.uuid(),
                 attributes: Dict = None, vector: List[float] = None):
        """
        Initialise document with content, id and attributes
        """
        if type(content) is not str:
            raise TypeError("content must be a string")

        if content == "":
            raise ValueError("content may not be the empty string ''")

        if type(id) is not str:
            raise TypeError("id must be a string")

        if id == "":
            raise ValueError("id may not be the empty string ''")

        self.id = id
        self.content = content
        self.attributes = attributes
        self.vector = vector


class ElasticDocument(Document):
    # def __init__(self, *args, **kwargs):
    #     super(ElasticDocument, self).__init__(*args, **kwargs)

    @staticmethod
    def elastic_mappings():
        """
        Get mappings for elastic index
        """
        dims = 512  # TODO: determine automatically
        mappings = {
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": dims
                },
                "content": {
                    "type": "text"
                },
                "attributes": {
                    "type": "object"
                }
            }
        }

        return mappings


class SearchResult:
    def __init__(self, document: Document, score: float, preview: str = None):
        self.document = document
        self.score = score
        self.preview = preview


class SearchResults:
    def __init__(self, max_score: float, num_results: int, results: List[SearchResult]):
        self.max_score = max_score
        self.num_results = num_results
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

    def search(self, query, vectorize_model, ids=None, body=None):
        """
        Search documents from elasticsearch
        """
        query_vec = vectorize_model.encode(query)

        if body is None:
            body = {
                # "size": n,
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
        hits = res.get("hits")

        max_score = hits.get("max_score")
        num_results = hits.get("total").get("value")

        search_results = []
        for hit in hits.get("hits"):
            # TODO: Separate to a function for json -> Document
            source = hit["_source"]
            score = hit["_score"]
            document = Document(source.get("content"), hit.get("_id"),
                                attributes=source.get("attributes"),
                                vector=source.get("vector"))
            search_result = SearchResult(document, score)
            search_results.append(search_result)

        return SearchResults(max_score, num_results, search_results)
