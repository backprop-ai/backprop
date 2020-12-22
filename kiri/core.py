from typing import Callable, Dict, List
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

from .search import DocStore, SearchResults, Document
from .utils import process_document, process_results
from .models import qa


class Kiri:
    """Core class of natural language engine

    Attributes:
        store: DocStore object to be used as the engine backend
        vectorize_model: Name of the SentenceTransformer model to be used in operations
        qa_model: Name of HuggingFace model to be used for Question/Answer
        process_doc_func: Function to be used when vectorizing updloaded documents
        process_results_func: Function to be used for calculating final scores of results

    Raises:
        ValueError: If a DocStore is not provided
    """

    def __init__(self, store: DocStore, vectorize_model: str = None,
                 qa_model: str = None,
                 process_doc_func: Callable[[
                     Document, SentenceTransformer], List[float]] = None,
                 process_results_func: Callable[[SearchResults, SentenceTransformer], None] = None):

        if store is None:
            raise ValueError("a DocStore implementation must be provided")

        if vectorize_model is None:
            # Use default vectorization model
            vectorize_model = "ojasaar/distilbert-sentence-msmarco-en-et"

        if process_doc_func is None:
            # Use default vectorizer
            process_doc_func = process_document

        if process_results_func is None:
            process_results_func = process_results

        self._store = store
        self._process_doc_func = process_doc_func
        self._process_results_func = process_results
        self._vectorize_model = SentenceTransformer(vectorize_model)

    def upload(self, documents: List[Document]) -> None:
        """Uploads documents to store

        Args:
            documents: List of documents for upload

        """

        return self._store.upload(documents, self._process_doc_func,
                                  self._vectorize_model)

    def search(self, query: str, max_results=10, min_score=0.0,
               preview_length=100, ids=None, body=None) -> SearchResults:
        """Search documents from document store

        Args:
            query: Search string on which search is performed
            max_results: Maximum amount of documents to be returned from search
            min_score: Minimum score required to be included in results
            preview_length: Number of characters in the preview/metatext of results
            ids: 
            body: Elasticsearch request body to be passed to the backend

        """
        search_results, query_vec = self._store.search(query, self._vectorize_model,
                                                       max_results=max_results, min_score=min_score,
                                                       ids=ids, body=body)
        self._process_results_func(
            search_results, query_vec, self._store._doc_class, preview_length)
        return search_results

    # TODO: "Modularize" this -- make separate QA components,
    #        rather than directly doing logic in here.
    #        Need default handling function (T5) w/ this logic,
    #        overridable with other function.
    def qa(self, query: str, context: str = None, context_doc: Document = None):
        """Perform QA, either on docstore or on provided context.

        """
        if context_doc or context:
            c_string = context if context else context_doc.content
            return qa(query, c_string)
        else:
            search_results = self.search(query)
            answers = []
            for result in search_results[:3]:
                text = result.document.content
                input_text = f"q: {query} c: {text}"
                features = self._qa_tokenizer(
                    [input_text], return_tensors="pt")
                output = self._qa_model.generate(input_ids=features["input_ids"],
                                                 attention_mask=features["attention_mask"])
                answer = self._qa_tokenizer.decode(output[0])
                answers.append(answer)
            return zip(answers, search_results[:3])
