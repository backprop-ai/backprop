import shortuuid
from typing import Dict, List


class Document:
    def __init__(self, content: str, id: str = None,
                 attributes: Dict = None, vector: List[float] = None):
        """
        Initialise document with content, id and attributes
        """
        if type(content) is not str:
            raise TypeError("content must be a string")

        if content == "":
            raise ValueError("content may not be the empty string ''")

        if id is None:
            id = shortuuid.uuid()

        if type(id) is not str:
            raise TypeError("id must be a string")

        if id == "":
            raise ValueError("id may not be the empty string ''")

        self.id = id
        self.content = content
        self.attributes = attributes
        self.vector = vector

    def to_json(self):
        return vars(self)


class ChunkedDocument(Document):
    def __init__(self, *args, chunking_level: int = 5, chunks: List[str] = None,
                 chunk_vectors: List[List[float]] = None, **kwargs):
        """
        Init chunked document, inheriting from document

        """
        super(ChunkedDocument, self).__init__(*args, **kwargs)
        if type(chunking_level) != int:
            raise TypeError("chunk_level must be an int")

        if chunking_level < 1:
            raise ValueError("chunk_level must be >= 1")

        if chunks and type(chunks) is not list:
            raise TypeError("chunks must be a list of strings")

        if chunk_vectors and type(chunk_vectors) is not list:
            raise TypeError("chunk_vectors must be a list of vectors")

        self.chunking_level = chunking_level
        self.chunks = chunks
        self.chunk_vectors = chunk_vectors

    def to_json(self):
        return vars(self)


class ElasticDocument(Document):
    @staticmethod
    def elastic_mappings(dims=768):
        """
        Get mappings for elastic index
        """
        # TODO: determine dims automatically
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

    def to_elastic(self):
        return vars(self)

    @classmethod
    def from_elastic(cls, *args, **kwargs):
        obj = cls.__new__(cls)

        super(ElasticDocument, obj).__init__(*args, **kwargs)
        return obj


class ElasticChunkedDocument(ChunkedDocument, ElasticDocument):
    def __init__(self, *args, **kwargs):
        super(ElasticChunkedDocument, self).__init__(*args, **kwargs)

    @staticmethod
    def elastic_mappings(dims=768):
        """
        Get mappings for elastic index
        """
        # TODO: determine dims automatically
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
                },
                "chunk_vectors": {
                    "type": "nested",
                    "properties": {
                        "vector": {
                            "type": "dense_vector",
                            "dims": dims
                        }
                    }
                }
            }
        }

        return mappings

    def to_elastic(self):
        json_repr = vars(self)
        json_repr["chunk_vectors"] = [{"vector": v}
                                      for v in json_repr["chunk_vectors"]]
        return json_repr

    @classmethod
    def from_elastic(cls, *args, **kwargs):
        obj = cls.__new__(cls)

        kwargs["chunk_vectors"] = [v.get("vector")
                                   for v in kwargs["chunk_vectors"]]

        super(ElasticChunkedDocument, obj).__init__(*args, **kwargs)
        return obj
