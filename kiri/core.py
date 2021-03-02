from typing import Callable, Dict, List, Tuple, Union
from kiri import save as utils_save
from kiri import load as utils_load
from kiri.tasks import QA, TextGeneration, TextVectorisation, Summarisation, Emotion, ImageClassification, TextClassification
from kiri.models import BaseModel

import logging


class Kiri:
    """Core class of kiri

    Attributes:
        vectorisation_model (optional): "english" or "multilingual".
            For local inference, the name of a SentenceTransformer model is also supported.
        classification_model (optional): "english" or "multilingual".
            For local inference, the name of a Huggingface transformers model is also supported.
        device (optional): Pytorch device to run inference on. Detected automatically if not specified.
    """

    def __init__(self, local=False, api_key=None,
                 text_vectorisation_model: Union[str, BaseModel] = None,
                 text_classification_model: Union[str, BaseModel] = None,
                 image_classification_model: Union[str, BaseModel] = None,
                 text_generation_model: Union[str, BaseModel] = None,
                 emotion_model: Union[str, BaseModel] = None,
                 summarisation_model: Union[str, BaseModel] = None,
                 qa_model: Union[str, BaseModel] = None,
                 device: str = None):

        if local and not device:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._vectorise = TextVectorisation(text_vectorisation_model, local=local,
                                        api_key=api_key, device=device, init=False)
        
        self._generate = TextGeneration(text_generation_model, local=local,
                                        api_key=api_key, device=device, init=False)

        self._classify = TextClassification(text_classification_model, local=local,
                            api_key=api_key, device=device, init=False)

        self._image_classification = ImageClassification(image_classification_model, local=local,
                            api_key=api_key, device=device, init=False)

        self._qa = QA(qa_model, local=local,
                            api_key=api_key, device=device, init=False)
        
        self._emotion = Emotion(emotion_model, local=local,
                            api_key=api_key, device=device, init=False)

        self._summarise = Summarisation(summarisation_model, local=local,
                            api_key=api_key, device=device, init=False)
        
        self._device = device
        self._local = local
        self._api_key = api_key


    # def upload(self, documents: List[Document]) -> None:
    #     """Processes and uploads documents to store

    #     Args:
    #         documents: List of documents for upload

    #     Returns:
    #         None

    #     Example:
    #         >>> kiri.upload([Document("First document"), Document("Second document")])
    #         None

    #     """
    #     logging.warning("Upload functionality is deprecated and will be removed in a future version. Use https://github.com/kiri-ai/kiri-search instead.")
    #     return self._store.upload(documents, self._process_doc_func)

    def save(self, model, path=None):
        return utils_save(model, path=path)

    def load(self, path):
        return utils_load(path)

    def search(self, query: str, max_results=10, min_score=0.0,
               preview_length=100, ids: List[str] = [], body=None):
        raise Exception("Search functionality is deprecated. Use https://github.com/kiri-ai/kiri-search instead.")

    def qa(self, question: str, context: str = None,
           prev_qa: List[Tuple[str, str]] = []):
        """Perform QA, either on docstore or on provided context.

        Args:
            question: Question (string or list of strings) for qa model.
            context: Context (string or list of strings) to ask question from.
            prev_qa (optional): List of previous question, answer tuples or list of prev_qa.

        Returns:
            Answer string or list of answer strings

        Example:
            >>> kiri.qa("Where does Sally live?", "Sally lives in London.")
            "London"
        """
        return self._qa(question, context, prev_qa=prev_qa)


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


    def image_classification(self, image_path: str, labels: List[str]):
        # TODO: Implement batching
        """Classify image according to given labels.


        Args:
            image_path: path to image
            labels: list of strings

        Returns:
            dict where each key is a label and value is probability between 0 and 1

        Example:
            >>> kiri.image_classification("/home/Documents/dog.png", ["cat", "dog"])
            {"cat": 0.01, "dog": 0.99}

        """
        return self._image_classification(image_path, labels)


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