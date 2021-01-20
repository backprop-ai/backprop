import shortuuid
from typing import Dict, List, Tuple


class Document:
    """Base document class for extension as needed.

    Attributes:
        content: Text content of the document
        id (optional): Unique ID for the document. Generated if not provided
        attributes (optional): Dictionary of user-defined attributes
        vector (optional): List of floats for doc vector representation

    Raises:
        TypeError: If content or id is not a string
        ValueError: If content or id is an empty string
    """

    def __init__(self, content: str, id: str = None,
                 attributes: Dict = None, vector: List[float] = None):
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
        self.doc_store = None

    def summarise(self):
        """Summarises document content"""
        # TODO: Handle long content
        return self.doc_store.kiri.summarise(self.content)

    def classify(self, labels: List[str]):
        """Classifies document content according to provided labels"""
        # TODO: Handle long content
        return self.doc_store.kiri.classify(self.content, labels)

    def qa(self, question, prev_qa: List[Tuple[str, str]] = []):
        """Performs qa on the document content"""
        # TODO: Handle long content
        return self.doc_store.kiri.qa(question, self.content, prev_qa=prev_qa)

    def emotion(self):
        """Detects emotion from document content"""
        # TODO: Handle long content
        return self.doc_store.kiri.emotion(self.content)

    def to_json(self, exclude_vectors=True):
        """Gets JSON form of the document.

        Args:
            exclude_vectors (optional): exclude vectors from json, defaults to True

        Returns:
            __dict__ of the document object

        """
        json_repr = vars(self).copy()
        del json_repr["doc_store"]
        if exclude_vectors:
            del json_repr["vector"]
        return json_repr


class ChunkedDocument(Document):
    """Document subclass with support for finer-grained vector chunking.

    Attributes:
        args: Document superclass arguments
        kwargs: Document superclass keyword arguments
        chunking_level (optional): Number of sentences in one chunk, defaults to 5
        chunks (optional): List of pre-chunked strings from the document
        chunk_vectors (optional): List of vectors for each chunk of the document

    Raises:
        TypeError: If chunk_level is not an int, 
            chunks is not a list of strings, or 
            chunk_vectors is not a list of vectors
        ValueError: If chunk_level is < 1

    """

    def __init__(self, *args, chunking_level: int = 5, chunks: List[str] = None,
                 chunk_vectors: List[List[float]] = None, **kwargs):
        super(ChunkedDocument, self).__init__(*args, **kwargs)
        if type(chunking_level) != int:
            raise TypeError("chunking_level must be an int")

        if chunking_level < 1:
            raise ValueError("chunking_level must be >= 1")

        if chunks and type(chunks) is not list:
            raise TypeError("chunks must be a list of strings")

        if chunk_vectors and type(chunk_vectors) is not list:
            raise TypeError("chunk_vectors must be a list of vectors")

        self.chunking_level = chunking_level
        self.chunks = chunks
        self.chunk_vectors = chunk_vectors

    def to_json(self, exclude_vectors=True):
        """Gets JSON form of the document
        Returns:
            __dict__ attr of the document object
        """
        json_repr = vars(self).copy()
        del json_repr["doc_store"]
        if exclude_vectors:
            del json_repr["vector"]
            del json_repr["chunk_vectors"]
        return json_repr


class ElasticDocument(Document):
    """Document with additional mapping required for ElasticSearch.

    Attributes:
        content: Text content of the document
        id: Unique ID for the document -- generated if not provided
        attributes: Dictionary of user-defined attributes
        vector: List of floats for doc vector representation
    """

    @staticmethod
    def elastic_mappings(dims=768):
        """Gets mappings for Elastic index

        Args:
            dims: Dimensions of document vector

        Returns:
            Dictionary of Elastic metadata 
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
        """Turns Document object into JSON form for Elastic

        Returns:
            __dict__ attribute of ElasticDocument object
        """
        json_repr = vars(self).copy()
        del json_repr["doc_store"]
        return json_repr

    @classmethod
    def from_elastic(cls, *args, **kwargs):
        obj = cls.__new__(cls)

        super(ElasticDocument, obj).__init__(*args, **kwargs)
        return obj


class ElasticChunkedDocument(ChunkedDocument, ElasticDocument):
    """Elastic-ready document with chunked vectorisation

    Attributes:
        args: Document & ChunkedDocument superclass arguments
    """

    def __init__(self, *args, **kwargs):
        super(ElasticChunkedDocument, self).__init__(*args, **kwargs)

    @staticmethod
    def elastic_mappings(dims=768):
        """Get mappings for elastic index

        Args:
            dims: Dimensions of document's vectors

        Returns:
            Dictionary of Elastic metadata 
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
        """Turns Document object into JSON form for Elastic

        Returns:
            __dict__ attribute of ElasticDocument object
        """
        json_repr = vars(self).copy()
        del json_repr["doc_store"]
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
