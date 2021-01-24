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
    
    def __call__(self, text, *args, **kwargs):
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