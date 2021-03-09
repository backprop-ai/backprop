from typing import Callable, Dict, List, Tuple, Union
from kiri import save as utils_save
from kiri import load as utils_load
from kiri import upload as utils_upload
from kiri.tasks import QA, TextGeneration, TextVectorisation, Summarisation, Emotion, ImageClassification, TextClassification
from kiri.models import BaseModel

import logging


class Kiri:
    """Core class of Kiri

    Attributes:
        text_vectorisation_model (optional): "english", "multilingual" or your own uploaded model.
            For local inference, a model class of instance TextVectorisationModel is also supported.
        text_classification_model (optional): "english", "multilingual" or your own uploaded model.
            For local inference, a model class of instance TextClassificationModel is also supported.
        image_classification_model (optional): "english" or your own uploaded model.
            For local inference, a model class of instance BaseModel is also supported.
        text_generation_model (optional): "gpt2", "t5-base-qa-summary-emotion" or your own uploaded model.
            For local inference, a model class of instance TextGenerationModel is also supported.
        emotion_model (optional): "english" or your own uploaded model.
            For local inference, a model class of instance BaseModel is also supported.
        summarisation_model (optional): "english" or your own uploaded model.
            For local inference, a model class of instance BaseModel is also supported.
        qa_model (optional): "english" or your own uploaded model.
            For local inference, a model class of instance BaseModel is also supported.
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

        if api_key == None:
            local = True

        if local and not device:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.TextVectorisation = TextVectorisation(text_vectorisation_model, local=local,
                                        api_key=api_key, device=device, init=False)
        
        self.TextGeneration = TextGeneration(text_generation_model, local=local,
                                        api_key=api_key, device=device, init=False)

        self.TextClassification = TextClassification(text_classification_model, local=local,
                            api_key=api_key, device=device, init=False)

        self.ImageClassification = ImageClassification(image_classification_model, local=local,
                            api_key=api_key, device=device, init=False)

        self.QA = QA(qa_model, local=local,
                            api_key=api_key, device=device, init=False)
        
        self.Emotion = Emotion(emotion_model, local=local,
                            api_key=api_key, device=device, init=False)

        self.Summarisation = Summarisation(summarisation_model, local=local,
                            api_key=api_key, device=device, init=False)
        
        self._device = device
        self._local = local
        self._api_key = api_key

    def save(self, model, path=None):
        """
        Saves the provided model to the kiri cache folder using model.name or to provided path

        Args:
            path: Optional path to save model
        """
        return utils_save(model, path=path)

    def load(self, path):
        """
        Loads a saved model and returns it.

        Args:
            path: Name of the model or full path to model.
        """
        return utils_load(path)

    def upload(self, model: BaseModel = None, path: str = None, save_path: str = None):
        """
        Deploys a model from object or path to Kiri. 
        Either the model or path to saved model must be provided.

        Args:
            model: Model object
            path: Path to saved model
            save_path: Optional path to save model if providing a model object
        """
        return utils_upload(model=model, path=path, save_path=save_path, api_key=self._api_key)

    def search(self, query: str, max_results=10, min_score=0.0,
               preview_length=100, ids: List[str] = [], body=None):
        """
        Search functionality is deprecated. Use https://github.com/kiri-ai/kiri-search instead.
        """
        raise Exception("Search functionality is deprecated. Use https://github.com/kiri-ai/kiri-search instead.")

    def qa(self, question: Union[str, List[str]], context: Union[str, List[str]],
            prev_qa: Union[List[Tuple[str, str]], List[List[Tuple[str, str]]]] = []):
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
        return self.QA(question, context, prev_qa=prev_qa)


    def summarise(self, text: Union[str, List[str]]):
        """Perform summarisation on input text.

        Args:
            input_text: string or list of strings to be summarised - keep each string below 500 words.

        Returns:
            Summary string or list of summary strings.

        Example:
            >>> kiri.summarise("This is a long document that contains plenty of words")
            "short summary of document"

        """
        return self.Summarisation(text)

    def emotion(self, text: Union[str, List[str]]):
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
        return self.Emotion(text)

    def classify(self, text: Union[str, List[str]], labels: Union[List[str], List[List[str]]]):
        """Renamed - use `classify_text` instead
        """
        logging.warning("classify has been renamed to classify_text, it will be removed in a future version")

        return self.TextClassification(text, labels)

    def classify_text(self, text: Union[str, List[str]], labels: Union[List[str], List[List[str]]]):
        """Classify input text according to given labels.


        Args:
            input_text: string or list of strings to classify
            labels: list of strings or list of labels

        Returns:
            dict where each key is a label and value is probability between 0 and 1, or list of dicts.

        Example:
            >>> kiri.classify_text("I am mad because my product broke.", ["product issue", "nature"])
            {"product issue": 0.98, "nature": 0.05}

        """
        return self.TextClassification(text, labels)


    def image_classification(self, image_path: Union[str, List[str]], labels: Union[List[str], List[List[str]]]):
        """Renamed - use `classify_image` instead
        """
        logging.warning("image_classification has been renamed to classify_image, it will be removed in a future version")
        return self.ImageClassification(image_path, labels)

    def classify_image(self, image_path: Union[str, List[str]], labels: Union[List[str], List[List[str]]]):
        """Classify image according to given labels.

        Args:
            image_path: path to image or list of paths to image
            labels: list of strings or list of labels

        Returns:
            dict where each key is a label and value is probability between 0 and 1 or list of dicts

        Example:
            >>> kiri.classify_image("/home/Documents/dog.png", ["cat", "dog"])
            {"cat": 0.01, "dog": 0.99}

        """
        return self.ImageClassification(image_path, labels)


    def vectorise(self, text: Union[str, List[str]]):
        """Renamed - use `vectorise_text` instead
        """
        logging.warning("vectorise has been renamed to vectorise_text, it will be removed in a future version")
        return self.TextVectorisation(text)

    def vectorise_text(self, text: Union[str, List[str]]):
        """Vectorise input text.


        Args:
            input_text: string or list of strings to vectorise

        Returns:
            Vector or list of vectors

        Example:
            >>> kiri.vectorise_text("iPhone 12 128GB")
            [0.92949192, 0.23123010, ...]

        """
        return self.TextVectorisation(text)

    def generate(self, text: Union[str, List[str]], min_length=10, max_length=20, temperature=1.0,
                top_k=0.0, top_p=1.0, repetition_penalty=1.0, length_penalty=1.0,
                num_beams=1, num_generations=1, do_sample=True):
        """Renamed - use `generate_text` instead
        """
        logging.warning("generate has been renamed to generate_text, it will be removed in a future version")
        return self.TextGeneration(text,
                          min_length=min_length,
                          max_length=max_length, temperature=temperature,
                          top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                          length_penalty=length_penalty, num_beams=num_beams,
                          num_generations=num_generations, do_sample=do_sample)

    
    def generate_text(self, text: Union[str, List[str]], min_length=10, max_length=20, temperature=1.0,
                top_k=0.0, top_p=1.0, repetition_penalty=1.0, length_penalty=1.0,
                num_beams=1, num_generations=1, do_sample=True):
        """Generates text to continue off the given input.

        Args:
            text: Text from which model will begin generating.
            min_length: Minimum length of generation before EOS can be generated.
            max_length: Maximum length of generated sequence.
            temperature: Value that alters softmax probabilities.
            top_k: Sampling strategy in which probabilities are redistributed among top k most-likely words.
            top_p: Sampling strategy in which probabilities are distributed among 
                set of words with combined probability greater than p.
            repetition_penalty: Penalty to be applied to words present in the text and
                words already generated in the sequence.
            length_penalty: Penalty applied to overall sequence length. Set >1 for longer sequences,
                or <1 for shorter ones. 
            num_beams: Number of beams to be used in beam search. (1: no beam search)
            num_generations: How many times to run generation. 
            do_sample: Whether or not sampling strategies (top_k & top_p) should be used.
        """
        return self.TextGeneration(text,
                          min_length=min_length,
                          max_length=max_length, temperature=temperature,
                          top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                          length_penalty=length_penalty, num_beams=num_beams,
                          num_generations=num_generations, do_sample=do_sample)