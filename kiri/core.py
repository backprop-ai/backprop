from typing import Callable, Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

from .search import DocStore, SearchResults, Document, InMemoryDocStore
from .utils import process_document, process_results
from .models import qa, summarise, emotion, zero_shot


class Kiri:
    """Core class of natural language engine

    Attributes:
        store: DocStore object to be used as the engine backend
        vectorize_model: Name of the SentenceTransformer model to be used in operations
        qa_model: Name of HuggingFace model to be used for Question/Answer
        process_doc_func: Function to be used when vectorizing updloaded documents
        process_results_func: Function to be used for calculating final scores of results

    """

    def __init__(self, store: DocStore = None, vectorize_model: str = None,
                 qa_model: str = None,
                 process_doc_func: Callable[[
                     Document, SentenceTransformer], List[float]] = None,
                 process_results_func: Callable[[SearchResults, SentenceTransformer], None] = None):

        if store is None:
            store = InMemoryDocStore()

        if vectorize_model is None:
            # Use default vectorization model
            vectorize_model = "msmarco-distilroberta-base-v2"

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
            search_results, query_vec, self._store._doc_class,
            preview_length, max_results, min_score)
        return search_results

    def qa(self, question: str, context: str = None,
           prev_qa: List[Tuple[str, str]] = [], context_doc: Document = None):
        """Perform QA, either on docstore or on provided context.

        """
        if context_doc or context:
            c_string = context if context else context_doc.content
            return qa(question, c_string, prev_qa=prev_qa)
        else:
            search_results = self.search(question)
            answers = []
            for result in search_results.results[:3]:
                c_string = result.document.content
                answer = qa(question, c_string, prev_qa=prev_qa)
                answers.append(answer)
            return list(zip(answers, search_results.results[:3]))

    def summarise(self, input_text):
        if type(input_text) != str:
            raise TypeError("input_text must be a string")

        if input_text == "":
            raise ValueError("input_text must not be an empty string")

        return summarise(input_text)

    def emotion(self, input_text):
        if type(input_text) != str:
            raise TypeError("input_text must be a string")

        if input_text == "":
            raise ValueError("input_text must not be an empty string")

        return emotion(input_text)

    def classify(self, input_text, labels: List[str]):
        if type(input_text) != str:
            raise TypeError("input_text must be a string")

        if input_text == "":
            raise ValueError("input_text must not be an empty string")

        if not isinstance(labels, list):
            raise TypeError("labels must be a list of strings")

        if len(labels) == 0:
            raise ValueError("labels must contain at least one label")

        return zero_shot(input_text, labels)
