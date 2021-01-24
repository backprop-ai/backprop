from typing import List, Tuple, Union
from ..models import BaseModel, GenerationModel
from .base import Task

import requests

DEFAULT_LOCAL_MODEL = "gpt2"

LOCAL_MODELS = {
    "gpt2": "gpt2",
    "t5-base-qa-summary-emotion": "t5-base-qa-summary-emotion"
}

DEFAULT_API_MODEL = "gpt2-large"

API_MODELS = ["gpt2-large", "t5-base-qa-summary-emotion"]

class Generation(Task):
    """
    Task for text generation.

    Attributes:
        model:
            1. Name of the model on Kiri's generation endpoint (gpt2-large, t5-base-qa-summary-emotion)
            2. Officially supported local models (gpt2, t5-base-qa-summary-emotion) or Huggingface path to the model.
            3. Kiri's GenerationModel object
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
                self.model = GenerationModel(self.model, init=init, device=device)

            task = getattr(self.model, "generate", None)
            if not callable(task):
                raise ValueError(f"The model {model} cannot be used for generation.\
                                It does not implement the 'generate' method.")
    
    def __call__(self, text: str, *args, **kwargs):
        """
        Calls the generation model with text and generation *args, **kwargs.
        """
        if self.local:
            return self.model.generate(text, *args, **kwargs)
        else:
            body = {
                "text": text,
                "model": self.model,
                **kwargs
            }

            res = requests.post("https://api.kiri.ai/generation", json=body,
                                headers={"x-api-key": self.api_key}).json()

            return res["output"]