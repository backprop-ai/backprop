from elasticsearch import Elasticsearch
from elasticsearch.helpers


class DocStore:
    def __init__(self, params):
        raise NotImplementedError("The constructor is not implemented!")

    def upload(self, params):
        raise NotImplementedError

    def search(self, params):
        raise NotImplementedError


class ElasticDocStore(DocStore):
    def __init__(self, url, index="kiri_default"):
        self._client = Elasticsearch([url])
        self._index = index

    def upload(self, documents, index=None):
        """
        Upload documents to elasticsearch
        """
        if not index:
            index = self._index

        print("index is", index)
        # self._client.bulk(documents)
        pass

    # def search(self, query, ids=None):
    #     """
    #     Search documents from elasticsearch
    #     """
    #     # self._client.search(index=self._index)
    #     pass


store = ElasticDocStore("a")
store.upload("<a", index="second_index")
