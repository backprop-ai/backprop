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
            1. Name of the model on Kiri's image classification endpoint (english or your own uploaded model)
            2. Officially supported local models (english) or Huggingface path to the model.
            3. Model class of instance Kiri's BaseModel that implements the image-classification task
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

    
    def __call__(self, image_path: Union[str, List[str]], labels: Union[List[str], List[List[str]]]):
        """Classify image according to given labels.

        Args:
            image_path: path to image or list of paths to image
            labels: list of strings or list of labels

        Returns:
            dict where each key is a label and value is probability between 0 and 1 or list of dicts
        """

        is_list = False

        if type(image_path) == list:
            is_list = True

        if not is_list:
            image_path = [image_path]

        image = []
        for img in image_path:
            with open(img, "rb") as image_file:
                img = base64.b64encode(image_file.read())
                image.append(img)

        if not is_list:
            image = image[0]
            
        if self.local:
            task_input = {
                "image": image,
                "labels": labels
            }
            return self.model(task_input, task="image-classification")
        else:
            body = {
                "image": image,
                "labels": labels,
                "model": self.model,
            }

            res = requests.post("https://api.kiri.ai/image-classification", json=body,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            return res["probabilities"]