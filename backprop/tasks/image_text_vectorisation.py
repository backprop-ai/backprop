from typing import List, Tuple, Union
from .base import Task
from backprop.models import CLIP, BaseModel
from backprop.utils import path_to_img, img_to_base64
import base64
from PIL import Image
from io import BytesIO
import torch

import requests

DEFAULT_LOCAL_MODEL = CLIP

LOCAL_MODELS = {
    "clip": DEFAULT_LOCAL_MODEL,
}

DEFAULT_API_MODEL = "clip"

API_MODELS = ["clip"]

FINETUNABLE_MODELS = []

class ImageTextVectorisation(Task):
    """
    Task for combined imag-text vectorisation.

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
    
    def __call__(self, image: Union[str, List[str]], text: Union[str, List[str]], return_tensor=False):
        """Vectorise input image and text pairs.

        Args:
            image: image or list of images to vectorise. Can be both PIL Image objects or paths to images.
            text: text or list of text to vectorise. Must match image ordering.

        Returns:
            Vector or list of vectors
        """
        vector = None
        image = path_to_img(image)

        if self.local:

            task_input = {
                "image": image,
                "text": text
            }
            vector = self.model(task_input, task="image-text-vectorisation",
                                return_tensor=return_tensor)
        else:
            raise NotImplementedError("This task is not yet implemented in the API")

            # image = img_to_base64(image)

            # body = {
            #     "image": image,
            #     "model": self.model
            # }

            # res = requests.post("https://api.backprop.co/image-vectorisation", json=body,
            #                     headers={"x-api-key": self.api_key}).json()

            # if res.get("message"):
            #     raise Exception(f"Failed to make API request: {res['message']}")

            # vector = res["vector"]

        if return_tensor and not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector)

        return vector

    # def finetune(self, *args, **kwargs):
    #     """
    #     Passes the args and kwargs to the model's finetune method.
    #     """
    #     try:
    #         return self.model.finetune(*args, **kwargs)
    #     except NotImplementedError:
    #         raise NotImplementedError(f"This model does not support finetuning, try: {', '.join(FINETUNABLE_MODELS)}")