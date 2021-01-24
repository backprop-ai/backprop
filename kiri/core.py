from typing import Callable, Dict, List, Tuple

from .search import DocStore, SearchResults, Document, InMemoryDocStore, ChunkedDocument
from .utils import process_documents, process_results
from .models import Generation, Vectorisation, Summarisation, Emotion, QA, Classification, T5QASummaryEmotion

import logging


class Kiri:
    """Core class of natural language engine

    Attributes:
        store (optional): DocStore object to be used as the engine backend
        vectorisation_model (optional): "english" or "multilingual".
            For local inference, the name of a SentenceTransformer model is also supported.
        classification_model (optional): "english" or "multilingual".
            For local inference, the name of a Huggingface transformers model is also supported.
        process_doc_func (optional): Function to be used when vectorising uploaded documents
        process_results_func (optional): Function to be used for calculating final scores of results
        device (optional): Pytorch device to run inference on. Detected automatically if not specified.
    """

    def __init__(self, store: DocStore = None, local=False, api_key=None,
                 vectorisation_model: str = None,
                 classification_model: str = None,
                 generation_model: str = None,
                 process_doc_func: Callable[[
                     Document, str], List[float]] = None,
                 process_results_func: Callable[[
                     SearchResults, str], None] = None,
                 device: str = None):

        if store is None:
            store = InMemoryDocStore()

        store.kiri = self

        if local and not device:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if process_doc_func is None:
            # Use default vectoriser
            process_doc_func = process_documents

        if process_results_func is None:
            process_results_func = process_results

        t5_qa_summary_emotion = T5QASummaryEmotion(device=device, init=False)

        self._vectorise = Vectorisation(vectorisation_model, local=local,
                                        api_key=api_key, device=device, init=False)
        
        self._generate = Generation(generation_model, local=local,
                                        api_key=api_key, device=device, init=False)

        self._classify = Classification(classification_model, local=local,
                            api_key=api_key, device=device, init=False)

        self._qa = QA(t5_qa_summary_emotion, local=local,
                            api_key=api_key, device=device, init=False)
        
        self._emotion = Emotion(t5_qa_summary_emotion, local=local,
                            api_key=api_key, device=device, init=False)

        self._summarise = Summarisation(t5_qa_summary_emotion, local=local,
                            api_key=api_key, device=device, init=False)
        
        self._device = device
        self._local = local
        self._api_key = api_key
        self._store = store
        self._process_doc_func = process_doc_func
        self._process_results_func = process_results

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
            return self._qa(question, context, prev_qa=prev_qa)
        else:
            search_results = self.search(question)
            answers = []
            if issubclass(self._store._doc_class, ChunkedDocument):
                for chunk in search_results.top_chunks[:num_answers]:
                    answer = self._qa(question, chunk["chunk"], prev_qa=prev_qa)
                    answers.append((answer, chunk["search_result"]))
            else:
                for result in search_results.results[:num_answers]:
                    c_string = result.document.content
                    answer = self._qa(question, c_string, prev_qa=prev_qa)
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
        return self._summarise(input_text)

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
        return self._emotion(input_text)

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
        return self._classify(input_text, labels)

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
        return self._vectorise(input_text)

    def generate(self, input_text, min_length=10, max_length=20, temperature=1.0,
                top_k=0.0, top_p=1.0, repetition_penalty=1.0, length_penalty=1.0,
                num_beams=1, num_generations=1, do_sample=True):
        """Generates text to continue off the given input.

        Args:
            input_text: Text from which model will begin generating.
            min_length: Minimum length of generation before EOS can be generated.
            max_length: Maximum length of generated sequence.
            temperature: Value that alters softmax probabilities.
            top_k: Sampling strategy in which probabilities are redistributed among top k most-likely words.
            top_p: Sampling strategy in which probabilities are distributed among 
                set of words with combined probability greater than p.
            repetition_penalty: Penalty to be applied to words present in the input_text and
                words already generated in the sequence.
            length_penalty: Penalty applied to overall sequence length. Set >1 for longer sequences,
                or <1 for shorter ones. 
            num_beams: Number of beams to be used in beam search. (1: no beam search)
            num_generations: How many times to run generation. 
            do_sample: Whether or not sampling strategies (top_k & top_p) should be used.
        """
        return self._generate(input_text,
                          min_length=min_length,
                          max_length=max_length, temperature=temperature,
                          top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                          length_penalty=length_penalty, num_beams=num_beams,
                          num_return_sequences=num_generations, do_sample=do_sample)