import shortuuid
from typing import Dict, List


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
