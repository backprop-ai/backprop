from typing import List, Tuple, Union
from ..models import BaseModel, ClassificationModel
from kiri.models import CLIP
from .base import Task
import base64

import requests

DEFAULT_LOCAL_MODEL = CLIP

LOCAL_MODELS = {
    "english": CLIP
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
            3. Model class of instance Kiri's BaseModel that implements the image-classification task
        local (optional): Run locally. Defaults to True
        api_key (optional): Kiri API key for non-local inference
        device (optional): Device to run inference on. Defaults to "cuda" if available.
        init (optional): Whether to initialise model immediately or wait until first call.
            Defaults to False
    """
    def __init__(self, model: Union[str, BaseModel] = None,
                local: bool = False, api_key: str = None, device: str = "cpu",
                init: bool = True):

        super().__init__(model, local=local, api_key=api_key, device=device,
                        init=init, local_models=LOCAL_MODELS, api_models=API_MODELS,
                        default_local_model=DEFAULT_LOCAL_MODEL,
                        default_api_model=DEFAULT_API_MODEL)

    
    def __call__(self, image_path: str, labels: List[str]):
        """Classify image according to given labels.

        Args:
            image_path: path to image
            labels: list of strings

        Returns:
            dict where each key is a label and value is probability between 0 and 1
        """
        if self.local:
            task_input = {
                "image": image_path,
                "labels": labels
            }
            return self.model(task_input, task="image-classification")
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