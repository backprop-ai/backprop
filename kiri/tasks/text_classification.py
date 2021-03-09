from typing import List, Tuple, Union
from kiri.models import BartLargeMNLI, XLMRLargeXNLI, BaseModel
from .base import Task

import requests

DEFAULT_LOCAL_MODEL = BartLargeMNLI

LOCAL_MODELS = {
    "english": DEFAULT_LOCAL_MODEL,
    "multilingual": XLMRLargeXNLI
}

DEFAULT_API_MODEL = "english"

API_MODELS = ["english", "multilingual"]

class TextClassification(Task):
    """
    Task for classification.

    Attributes:
        model:
            1. Name of the model on Kiri's classification endpoint (english, multilingual or your own uploaded model)
            2. Officially supported local models (english, multilingual).
            3. Model class of instance Kiri's TextClassificationModel
            4. Path/name of saved Kiri model
        local (optional): Run locally. Defaults to False
        api_key (optional): Kiri API key for non-local inference
        device (optional): Device to run inference on. Defaults to "cuda" if available.
        init (optional): Whether to initialise model immediately or wait until first call.
            Defaults to True
    """
    def __init__(self, model: Union[str, BaseModel] = None,
                local: bool = False, api_key: str = None, device: str = "cpu",
                init: bool = True):

        super().__init__(model, local=local, api_key=api_key, device=device,
                        init=init, local_models=LOCAL_MODELS, api_models=API_MODELS,
                        default_local_model=DEFAULT_LOCAL_MODEL,
                        default_api_model=DEFAULT_API_MODEL)

    
    def __call__(self, text: Union[str, List[str]], labels: Union[List[str], List[List[str]]]):
        """Classify input text according to given labels.

        Args:
            text: string or list of strings to classify
            labels: list of strings or list of labels

        Returns:
            dict where each key is a label and value is probability between 0 and 1, or list of dicts.
        """
        if self.local:
            task = "text-classification"
            task_input = {
                "text": text,
                "labels": labels
            }
            return self.model(task_input, task=task)
        else:
            body = {
                "text": text,
                "labels": labels,
                "model": self.model,
            }

            res = requests.post("https://api.kiri.ai/text-classification", json=body,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            return res["probabilities"]