import time
from .documents import Document, ElasticDocument, ChunkedDocument
from elasticsearch import Elasticsearch, helpers
from typing import Dict, List
import shortuuid
from ..utils import elastic_to_search_results, check_duplicate_documents, \
    check_document_types, batch_items


class SearchResult:
    """Basic Kiri search result

    Attributes:
        document: Document object associated with result
        score: Float representing relevancy to search query
        preview: Text content "peek" at result (metatext)
    """

    def __init__(self, document: Document, score: float, preview: str = None):
        self.document = document
        self.score = score
        self.preview = preview

    def to_json(self, exclude_vectors=True):
        """Gets JSON form of search result

        Attributes:
            exclude_vectors: Exclude content and chunk vectors from the json output

        Returns:
            __dict__ attribute of SearchResult object
        """
        json_repr = vars(self)
        json_repr["document"] = json_repr["document"].to_json(
            exclude_vectors=exclude_vectors)

        return json_repr


class SearchResults:
    """List of SearchResults with search metadata

    Attributes:
        max_score: Highest score returned by the search
        total_results: Count of documents found
        results: List of <total_results> SearchResult objects
    """

    def __init__(self, max_score: float, total_results: int, results: List[SearchResult]):
        self.max_score = max_score
        self.total_results = total_results
        self.results = results
        self.top_chunks = []

    def to_json(self, exclude_vectors=True):
        """Gets JSON form of returned results

        Attributes:
            exclude_vectors: Exclude content and chunk vectors from the json output

        Returns:
            __dict__ attribute of SearchResults object

        """
        json_repr = vars(self)
        json_repr["results"] = [
            r.to_json(exclude_vectors=exclude_vectors) for r in json_repr["results"]]
        return json_repr


class DocStore:
    """Base DocStore class for extension.

    Raises:
        NotImplementedError: If core init, upload, or search functions are not implemented
    """

    def __init__(self, *args, **kwargs):
        self.kiri = None
        raise NotImplementedError("__init__ is not implemented!")

    def upload(self, *args, **kwargs):
        raise NotImplementedError("upload is not implemented!")

    def search(self, *args, **kwargs):
        raise NotImplementedError("search is not implemented!")


class InMemoryDocStore(DocStore):
    """DocStore variant for in memory document storage.
    Meant to be used for development and testing.

    Attributes:
        doc_class (optional): Document class used for the store, defaults to Document 
    """

    def __init__(self, doc_class: Document = Document):
        self.documents = []
        self._doc_class = doc_class

    def upload(self, documents: List[Document], vectorise_func) -> None:
        """Process and upload documents to memory

        Args:
            documents: List of documents
            vectorise_func: Function used for vectorisation
            vectorise_model: SentenceTransformer model used for vectorisation
        """

        # Add doc_store to documents
        for d in documents:
            d.doc_store = self
        # Check ID uniqueness
        check_duplicate_documents(documents)
        # Check type consistency
        check_document_types(documents)
        # Batching
        batches = batch_items(documents)

        # Update document class conveniently
        if issubclass(type(documents[0]), ChunkedDocument):
            self._doc_class = ChunkedDocument

        for batch in batches:
            vectorise_func(batch, self)
            self.documents += batch

    def search(self, query, max_results=10, min_score=0.0,
               ids=None, body=None):
        query_vec = self.kiri.vectorise(query)

        documents = self.documents

        # Filter by id
        if ids:
            documents = [d for d in documents if d.id in ids]

        results = [SearchResult(d, -1.0) for d in documents]
        search_results = SearchResults(-1.0, len(results), results)
        return search_results, query_vec


class ElasticDocStore(DocStore):
    """DocStore variant for an Elasticsearch backend.

    Attributes:
        url: Location at which Elasticsearch is running
        index: Elastic index (database) to be used for this doc store
        doc_class: Document object type being stored in this index
    """

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

    def upload(self, documents: List[ElasticDocument], vectorise_func, index: str = None) -> None:
        """Upload documents to Elasticsearch

        Args:
            documents: List of documents to be uploaded to backend
            vectorise_func: Function used to vectorise document contents
            index: Index (db) to be used -- uses initialized default if none provided
        """
        if not index:
            index = self._index

        # Add doc_store to documents
        for d in documents:
            d.doc_store = self
        # Check ID uniqueness
        check_duplicate_documents(documents)
        # Check type consistency
        check_document_types(documents)
        # Batching
        batches = batch_items(documents)

        for batch in batches:
            payload = []
            # Calculate vectors
            vectorise_func(batch, self)

            for document in batch:
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

    def search(self, query, max_results=10, min_score=0.0, ids: List[str] = [], body=None):
        """Search documents from Elasticsearch

        Args:
            query: Question from which search is based
            max_results: Maximum number of search results to return
            min_score: Minimum relevancy score required to be included in results
            ids: Filter search results by ids
            body: Elasticsearch parameters to be used in search -- default made if none provided

        Returns:
            Tuple with the SearchResults object and the vectorised query
        """
        query_vec = self.kiri.vectorise(query)

        # elasticsearch does not support negative scores
        score_modifier = 1.0

        if body is None:
            q = {
                "bool": {
                    "should": [
                        {"match": {
                            "content": query
                        }},
                        {"match_all": {}}

                    ]
                }
            }

            if ids:
                q = {
                    "bool": {
                        "must": [
                            {
                                "terms": {
                                    "_id": ids
                                }
                            },
                            q
                        ]
                    }
                }

            body = {
                "min_score": min_score + score_modifier,
                "size": max_results,
                "query": {
                    "function_score": {
                        "query": q,
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

        # Add doc_store to documents
        for result in search_results.results:
            result.document.doc_store = self

        return search_results, query_vec
