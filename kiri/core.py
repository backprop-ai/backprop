from typing import Callable, Dict, List, Tuple

from .search import DocStore, SearchResults, Document, InMemoryDocStore, ChunkedDocument
from .utils import process_documents, process_results
from .models import qa, summarise, emotion, zero_shot, vectorise

import logging


class Kiri:
    """Core class of natural language engine

    Attributes:
        store: DocStore object to be used as the engine backend
        vectorise_model: Name of the SentenceTransformer model to be used in operations
        process_doc_func: Function to be used when vectorising uploaded documents
        process_results_func: Function to be used for calculating final scores of results
        device: Pytorch device to run inference on. Detected automatically if not specified.
    """

    def __init__(self, store: DocStore = None, local=False, api_key=None,
                 vectorise_model: str = None,
                 process_doc_func: Callable[[
                     Document, str], List[float]] = None,
                 process_results_func: Callable[[
                     SearchResults, str], None] = None,
                 device: str = None):

        if store is None:
            store = InMemoryDocStore()

        store.kiri = self

        if local == False and api_key == None:
            raise ValueError(
                "Please provide your api_key (https://kiri.ai) with api_key=... or set local=True")

        if local == False and vectorise_model:
            logging.warning(
                "User specified models for non-local inference are not supported. Using default models.")

        if local and not device:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if process_doc_func is None:
            # Use default vectoriser
            process_doc_func = process_documents

        if process_results_func is None:
            process_results_func = process_results

        self._device = device
        self._local = local
        self._api_key = api_key
        self._store = store
        self._process_doc_func = process_doc_func
        self._process_results_func = process_results
        self._vectorise_model = vectorise_model

    def upload(self, documents: List[Document]) -> None:
        """Processes and uploads documents to store

        Args:
            documents: List of documents for upload

        Returns:
            None

        Example:
            >>> kiri.upload([Document("First document"), Document("Second document")])
            None

        """

        return self._store.upload(documents, self._process_doc_func)

    def search(self, query: str, max_results=10, min_score=0.0,
               preview_length=100, ids: List[str] = [], body=None) -> SearchResults:
        """Search documents from document store

        Args:
            query: Search string on which search is performed
            max_results: Maximum amount of documents to be returned from search
            min_score: Minimum score required to be included in results
            preview_length: Number of characters in the preview/metatext of results
            ids: List of ids to search from
            body: Elasticsearch request body to be passed to the backend

        Returns:
            SearchResults object

        Example:
            >>> kiri.search("RTX 3090")
            SearchResults object

        """
        search_results, query_vec = self._store.search(query,
                                                       max_results=max_results, min_score=min_score,
                                                       ids=ids, body=body)
        self._process_results_func(
            search_results, query_vec, self._store._doc_class,
            preview_length, max_results, min_score)
        return search_results

    def qa(self, question: str, context: str = None,
           prev_qa: List[Tuple[str, str]] = [], num_answers: int = 3):
        """Perform QA, either on docstore or on provided context.

        Args:
            question: Question (string or list of strings if using own context) for qa model.
            context (optional): Context (string or list of strings) to ask question from.
            prev_qa (optional): List of previous question, answer tuples or list of prev_qa.
            num_answers (optional): Number of answers to return

        Returns:
            if context is given: Answer string or list of answer strings
            if no context: List of num_answers (default 3) answer and SearchResult object pairs.

        Example:
            >>> kiri.qa("Where does Sally live?", "Sally lives in London.")
            "London"
        """
        if context:
            return qa(question, context, prev_qa=prev_qa,
                      local=self._local, api_key=self._api_key, device=self._device)
        else:
            search_results = self.search(question)
            answers = []
            if issubclass(self._store._doc_class, ChunkedDocument):
                for chunk in search_results.top_chunks[:num_answers]:
                    answer = qa(question, chunk["chunk"], prev_qa=prev_qa,
                                local=self._local, api_key=self._api_key, device=self._device)
                    answers.append((answer, chunk["search_result"]))
            else:
                for result in search_results.results[:num_answers]:
                    c_string = result.document.content
                    answer = qa(question, c_string, prev_qa=prev_qa,
                                local=self._local, api_key=self._api_key, device=self._device)
                    answers.append((answer, result))
            return answers

    def summarise(self, input_text):
        """Perform summarisation on input text.

        Args:
            input_text: string or list of strings to be summarised - keep each string below 500 words.

        Returns:
            Summary string or list of summary strings.

        Example:
            >>> kiri.summarise("This is a long document that contains plenty of words")
            "short summary of document"

        """
        # if type(input_text) != str:
        #     raise TypeError("input_text must be a string")

        # if input_text == "":
        #     raise ValueError("input_text must not be an empty string")

        return summarise(input_text, local=self._local, api_key=self._api_key, device=self._device)

    def emotion(self, input_text):
        """Perform emotion detection on input text.

        Args:
            input_text: string or list of strings to detect emotion from
                keep this under a few sentences for best performance.

        Returns:
            Emotion string or list of emotion strings.
                Each emotion string contains comma and space separated emotions.
            Emotions are from: admiration, approval, annoyance, gratitude, disapproval, amusement,
                curiosity, love, optimism, disappointment, joy, realization, anger, sadness, confusion,
                caring, excitement, surprise, disgust, desire, fear, remorse, embarrassment, nervousness,
                pride, relief, grief

        Example:
            >>> kiri.emotion("I really like what you did there")
            "approval"

        """
        # if type(input_text) != str:
        #     raise TypeError("input_text must be a string")

        # if input_text == "":
        #     raise ValueError("input_text must not be an empty string")

        return emotion(input_text, local=self._local, api_key=self._api_key, device=self._device)

    def classify(self, input_text, labels: List[str]):
        """Classify input text according to given labels.


        Args:
            input_text: string or list of strings to classify
            labels: list of strings or list of labels

        Returns:
            dict where each key is a label and value is probability between 0 and 1, or list of dicts.

        Example:
            >>> kiri.classify("I am mad because my product broke.", ["product issue", "nature"])
            {"product issue": 0.98, "nature": 0.05}

        """
        # if type(input_text) != str:
        #     raise TypeError("input_text must be a string")

        # if input_text == "":
        #     raise ValueError("input_text must not be an empty string")

        # if not isinstance(labels, list):
        #     raise TypeError("labels must be a list of strings")

        # if len(labels) == 0:
        #     raise ValueError("labels must contain at least one label")

        return zero_shot(input_text, labels,
                         local=self._local, api_key=self._api_key, device=self._device)

    def vectorise(self, input_text):
        """Vectorise input text.


        Args:
            input_text: string or list of strings to vectorise

        Returns:
            Vector or list of vectors

        Example:
            >>> kiri.vectorise("iPhone 12 128GB")
            [0.92949192, 0.23123010, ...]

        """
        # if type(input_text) != str:
        #     raise TypeError("input_text must be a string")

        # if input_text == "":
        #     raise ValueError("input_text must not be an empty string")

        # if not isinstance(labels, list):
        #     raise TypeError("labels must be a list of strings")

        # if len(labels) == 0:
        #     raise ValueError("labels must contain at least one label")

        return vectorise(input_text, model_name=self._vectorise_model,
                         local=self._local, api_key=self._api_key, device=self._device)
