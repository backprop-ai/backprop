from typing import List, Tuple, Union
from .base import Task
from backprop.models import MSMARCODistilrobertaBaseV2, DistiluseBaseMultilingualCasedV2, CLIP, BaseModel

import requests

DEFAULT_LOCAL_MODEL = MSMARCODistilrobertaBaseV2

LOCAL_MODELS = {
    "english": DEFAULT_LOCAL_MODEL,
    "multilingual": DistiluseBaseMultilingualCasedV2,
    "clip": CLIP
}

DEFAULT_API_MODEL = "english"

API_MODELS = ["english", "multilingual", "clip"]

FINETUNABLE_MODELS = ["english", "multilingual"]

class TextVectorisation(Task):
    """
    Task for text vectorisation.

    Attributes:
        model:
            1. Name of the model on Backprop's vectorisation endpoint (english, multilingual, clip or your own uploaded model)
            2. Officially supported local models (english, multilingual, clip).
            3. Model class of instance Backprop's TextVectorisationModel
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
    
    def __call__(self, text: Union[str, List[str]]):
        """Vectorise input text.

        Args:
            text: string or list of strings to vectorise. Can be both PIL Image objects or paths to images.

        Returns:
            Vector or list of vectors
        """
        if self.local:
            task_input = {
                "text": text
            }
            return self.model(task_input, task="text-vectorisation")
        else:
            body = {
                "text": text,
                "model": self.model
            }

            res = requests.post("https://api.backprop.co/text-vectorisation", json=body,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            return res["vector"]

    def finetune(self, *args, **kwargs):
        """
        Passes the args and kwargs to the model's finetune method.
        """
        try:
            return self.model.finetune(*args, **kwargs)
        except NotImplementedError:
            raise NotImplementedError(f"This model does not support finetuning, try: {', '.join(FINETUNABLE_MODELS)}")