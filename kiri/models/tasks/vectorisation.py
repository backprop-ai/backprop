from typing import List, Tuple, Union
from ..models import BaseModel, VectorisationModel
from .base import Task

import requests

DEFAULT_LOCAL_MODEL = "msmarco-distilroberta-base-v2"

LOCAL_MODELS = {
    "english": DEFAULT_LOCAL_MODEL,
    "multilingual": "distiluse-base-multilingual-cased-v2"
}

DEFAULT_API_MODEL = "english"

API_MODELS = ["english", "multilingual"]

class Vectorisation(Task):
    """
    Task for text vectorisation.

    Attributes:
        model:
            1. Name of the model on Kiri's vectorisation endpoint (english, multilingual)
            2. Officially supported local models (english, multilingual) or Sentence Transformer path to the model.
            3. Kiri's VectorisationModel object
        local (optional): Run locally. Defaults to True
        api_key (optional): Kiri API key for non-local inference
        device (optional): Device to run inference on. Defaults to "cuda" if available.
        init (optional): Whether to initialise model immediately or wait until first call.
            Defaults to False
    """
    def __init__(self, model: Union[str, BaseModel] = None,
                local: bool = False, api_key: str = None, device: str = "cpu",
                init: bool = False):

        super().__init__(model, local=local, api_key=api_key, device=device,
                        init=init, local_models=LOCAL_MODELS, api_models=API_MODELS,
                        default_local_model=DEFAULT_LOCAL_MODEL,
                        default_api_model=DEFAULT_API_MODEL)
        
        # Model must implement the task
        if self.local:
            # Still needs to be initialised
            if type(self.model) == str:
                self.model = VectorisationModel(self.model, init=init, device=device)

            task = getattr(self.model, "vectorise", None)
            if not callable(task):
                raise ValueError(f"The model {model} cannot be used for vectorisation.\
                                It does not implement the `vectorise` method.")
    
    def __call__(self, text, *args, **kwargs):
        """
        Calls the vectorisation model with text and vectorisation *args, **kwargs.
        """
        if self.local:
            return self.model.vectorise(text, *args, **kwargs)
        else:
            body = {
                "text": text,
                "model": self.model
            }

            res = requests.post("https://api.kiri.ai/vectorisation", json=body,
                                headers={"x-api-key": self.api_key}).json()

            return res["vector"]