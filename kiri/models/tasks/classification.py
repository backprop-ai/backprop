from typing import List, Tuple, Union
from ..models import BaseModel, ClassificationModel
from .base import Task

import requests

DEFAULT_LOCAL_MODEL = "facebook/bart-large-mnli"

LOCAL_MODELS = {
    "english": DEFAULT_LOCAL_MODEL,
    "multilingual": "joeddav/xlm-roberta-large-xnli"
}

DEFAULT_API_MODEL = "english"

API_MODELS = ["english", "multilingual"]

class Classification(Task):
    """
    Task for classification.

    Attributes:
        model:
            1. Name of the model on Kiri's classification endpoint (english, multilingual)
            2. Officially supported local models (english, multilingual) or Huggingface path to the model.
            3. Kiri's ClassificationModel object
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
                self.model = ClassificationModel(self.model, init=init, device=device)

            task = getattr(self.model, "classify", None)
            if not callable(task):
                raise ValueError(f"The model {model} cannot be used for classification.\
                                It does not implement the 'classify' method.")
    
    def __call__(self, text: str, labels: List[str]):
        """
        Calls the classification model with text and labels.
        """
        if self.local:
            return self.model.classify(text, labels)
        else:
            body = {
                "text": text,
                "labels": labels,
                "model": self.model,
            }

            res = requests.post("https://api.kiri.ai/classification", json=body,
                                headers={"x-api-key": self.api_key}).json()

            return res["probabilities"]