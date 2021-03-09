from typing import List, Tuple, Union
from .base import Task
from kiri.models import MSMARCODistilrobertaBaseV2, DistiluseBaseMultilingualCasedV2, BaseModel

import requests

DEFAULT_LOCAL_MODEL = MSMARCODistilrobertaBaseV2

LOCAL_MODELS = {
    "english": DEFAULT_LOCAL_MODEL,
    "multilingual": DistiluseBaseMultilingualCasedV2
}

DEFAULT_API_MODEL = "english"

API_MODELS = ["english", "multilingual"]

class TextVectorisation(Task):
    """
    Task for text vectorisation.

    Attributes:
        model:
            1. Name of the model on Kiri's vectorisation endpoint (english, multilingual or your own uploaded model)
            2. Officially supported local models (english, multilingual).
            3. Model class of instance Kiri's TextVectorisationModel
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
    
    def __call__(self, text: Union[str, List[str]]):
        """Vectorise input text.

        Args:
            text: string or list of strings to vectorise

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

            res = requests.post("https://api.kiri.ai/text-vectorisation", json=body,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            return res["vector"]