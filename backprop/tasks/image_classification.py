from typing import List, Tuple, Union
from ..models import BaseModel, ClassificationModel
from backprop.models import CLIP, EfficientNet
from .base import Task
from PIL import Image
import base64

import requests
from io import BytesIO

DEFAULT_LOCAL_MODEL = CLIP

LOCAL_MODELS = {
    "english": CLIP,
    "efficientnet": EfficientNet
}

DEFAULT_API_MODEL = "english"
FINETUNABLE_MODELS = ["efficientnet"]

API_MODELS = ["english"]

class ImageClassification(Task):
    """
    Task for classification.

    Attributes:
        model:
            1. Name of the model on Backprop's image classification endpoint (english, efficientnet or your own uploaded model)
            2. Officially supported local models (english, efficientnet).
            3. Model class of instance Backprop's BaseModel that implements the image-classification task
            4. Path/name of saved Backprop model
        local (optional): Run locally. Defaults to False
        api_key (optional): Backprop API key for non-local inference
        device (optional): Device to run inference on. Defaults to "cuda" if available.
    """
    def __init__(self, model: Union[str, BaseModel] = None,
                local: bool = False, api_key: str = None, device: str = None):

        super().__init__(model, local=local, api_key=api_key, device=device,
                        local_models=LOCAL_MODELS, api_models=API_MODELS,
                        default_local_model=DEFAULT_LOCAL_MODEL,
                        default_api_model=DEFAULT_API_MODEL)

    
    def __call__(self, image_path: Union[str, List[str]], labels: Union[List[str], List[List[str]]] = None):
        """Classify image according to given labels.

        Args:
            image_path: path to image or list of paths to image
            labels: list of strings or list of labels (for zero shot classification)

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
            if not isinstance(img, Image.Image):
                with open(img, "rb") as image_file:
                    img = base64.b64encode(image_file.read())
            else:
                buffered = BytesIO()
                img.save(buffered, format=img.format)
                img = base64.b64encode(buffered.getvalue())
            
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

            res = requests.post("https://api.backprop.co/image-classification", json=body,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            return res["probabilities"]

    def finetune(self, *args, **kwargs):
        """
        Passes the args and kwargs to the model's finetune method.
        """
        try:
            return self.model.finetune(*args, **kwargs)
        except NotImplementedError:
            raise NotImplementedError(f"This model does not support finetuning, try: {', '.join(FINETUNABLE_MODELS)}")