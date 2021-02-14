from typing import List, Tuple, Union
from ..models import BaseModel, ClassificationModel
from ..custom_models import CLIP
from .base import Task
import base64

import requests

DEFAULT_LOCAL_MODEL = "CLIP"

LOCAL_MODELS = {
    "english": DEFAULT_LOCAL_MODEL
}

DEFAULT_API_MODEL = "english"

API_MODELS = ["english"]

class ImageClassification(Task):
    """
    Task for classification.

    Attributes:
        model:
            1. Name of the model on Kiri's image classification endpoint (english)
            2. Officially supported local models (english) or Huggingface path to the model.
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
                if self.model == "CLIP":
                    self.model = CLIP(init=init, device=device)

            task = getattr(self.model, "image_classification", None)
            if not callable(task):
                raise ValueError(f"The model {model} cannot be used for classification.\
                                It does not implement the 'image_classification' method.")
    
    def __call__(self, image_path: str, labels: List[str]):
        """
        Calls the classification model with text and labels.
        """
        if self.local:
            return self.model.image_classification(image_path, labels)
        else:
            # Base64 encode
            with open(image_path, "rb") as image_file:
                image = base64.b64encode(image_file.read())
            body = {
                "image": image,
                "labels": labels,
                "model": self.model,
            }

            res = requests.post("https://api.kiri.ai/image-classification", json=body,
                                headers={"x-api-key": self.api_key}).json()

            return res["probabilities"]