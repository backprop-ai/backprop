from elasticsearch import Elasticsearch, helpers
from typing import Dict
import shortuuid


class Document:
    def __init__(self, content: str, id: str = shortuuid.uuid(), attributes: Dict = None):
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

        self._id = id
        self._content = content
        self._attributes = attributes
        self._vectors = None


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
                "_vectors": {
                    "type": "dense_vector",
                    "dims": dims
                },
                "_content": {
                    "type": "text"
                },
                "_attributes": {
                    "type": "object"
                }
            }
        }

        return mappings


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

    def process(self, document: Document):
        """
        Process and update missing document fields (vectors)
        """

    def upload(self, documents: [Document], vectorize_func, vectorize_model, index: str = None) -> None:
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
            if document._vectors is None:
                document._vectors = vectorize_func(document, vectorize_model)

            # JSON representation of document
            doc_json = document.__dict__

            # Add correct index
            doc_json["_index"] = index

            payload.append(doc_json)

        # Bulk upload to elasticsearch
        helpers.bulk(self._client, payload)

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
                            "source": "cosineSimilarity(params.query_vector, '_vectors') + 1.0",
                            "params": {
                                "query_vector": query_vec.tolist()}
                        }

                    },
                }
                # "aggs": {
                #     "types_count": {"value_count": {"field": "url"}}
                # }
            }

        return self._client.search(index=self._index, body=body)
