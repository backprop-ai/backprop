from typing import List, Tuple, Union
from .base import Task
from kiri.models import T5QASummaryEmotion, BaseModel

import requests

DEFAULT_LOCAL_MODEL = T5QASummaryEmotion

LOCAL_MODELS = {
    "english": DEFAULT_LOCAL_MODEL
}

DEFAULT_API_MODEL = "english"

API_MODELS = ["english"]

class Emotion(Task):
    """
    Task for emotion detection.

    Attributes:
        model:
            1. Name of the model on Kiri's emotion endpoint (english or your own uploaded model)
            2. Officially supported local models (english).
            3. Model class of instance Kiri's BaseModel that implements the emotion task
            4. Path/name of saved Kiri model
        model_class (optional): The model class to use when supplying a path for the model.
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
        """Perform emotion detection on input text.

        Args:
            text: string or list of strings to detect emotion from
                keep this under a few sentences for best performance.

        Returns:
            Emotion string or list of emotion strings.
        """
        task_input = {
            "text": text
        }
        if self.local:
            return self.model(task_input, task="emotion")
        else:
            task_input["model"] = self.model

            res = requests.post("https://api.kiri.ai/emotion", json=task_input,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            return res["emotion"]