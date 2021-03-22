from typing import List, Tuple, Union
from .base import Task
from backprop.models import CLIP, BaseModel
import base64
from PIL import Image
from io import BytesIO

import requests

DEFAULT_LOCAL_MODEL = CLIP

LOCAL_MODELS = {
    "clip": DEFAULT_LOCAL_MODEL,
}

DEFAULT_API_MODEL = "clip"

API_MODELS = ["clip"]

FINETUNABLE_MODELS = []

class ImageVectorisation(Task):
    """
    Task for text vectorisation.

    Attributes:
        model:
            1. Name of the model on Backprop's vectorisation endpoint (clip or your own uploaded model)
            2. Officially supported local models (clip).
            3. Model class of instance Backprop's BaseModel (that supports the task)
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
    
    def __call__(self, image_path: Union[str, List[str]]):
        """Vectorise input image.

        Args:
            text: image or list of images to vectorise. Can be both PIL Image objects or paths to images.

        Returns:
            Vector or list of vectors
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
                "image": image
            }
            return self.model(task_input, task="image-vectorisation")
        else:
            body = {
                "image": image,
                "model": self.model
            }

            res = requests.post("https://api.backprop.co/image-vectorisation", json=body,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            return res["vector"]

    # def finetune(self, *args, **kwargs):
    #     """
    #     Passes the args and kwargs to the model's finetune method.
    #     """
    #     try:
    #         return self.model.finetune(*args, **kwargs)
    #     except NotImplementedError:
    #         raise NotImplementedError(f"This model does not support finetuning, try: {', '.join(FINETUNABLE_MODELS)}")