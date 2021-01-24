from typing import List, Tuple, Union
from ..models import BaseModel
from ..custom_models import T5QASummaryEmotion
from .base import Task

import requests

DEFAULT_LOCAL_MODEL = "kiri-ai/t5-base-qa-summary-emotion"

LOCAL_MODELS = {
    "english": DEFAULT_LOCAL_MODEL
}

DEFAULT_API_MODEL = "english"

API_MODELS = ["english"]

class QA(Task):
    """
    Task for Question Answering.

    Attributes:
        model:
            1. Name of the model on Kiri's qa endpoint (english)
            2. Officially supported local models (english) or Huggingface path to the model.
            3. Kiri's BaseModel object that implements the qa method
        model_class (optional): The model class to use when supplying a path for the model.
        local (optional): Run locally. Defaults to True
        api_key (optional): Kiri API key for non-local inference
        device (optional): Device to run inference on. Defaults to "cuda" if available.
        init (optional): Whether to initialise model immediately or wait until first call.
            Defaults to False
    """
    def __init__(self, model: Union[str, BaseModel] = None, model_class=T5QASummaryEmotion,
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
                self.model = model_class(self.model, init=init, device=device)

            task = getattr(self.model, "qa", None)
            if not callable(task):
                raise ValueError(f"The model {model} cannot be used for qa.\
                                It does not implement the 'qa' method.")
    
    def __call__(self, question: str, context: str, prev_qa: List[Tuple[str, str]] = []):
        """
        Calls the qa model with question, context and list of previous question, answer pairs.
        """
        if self.local:
            return self.model.qa(question, context, prev_qa=prev_qa)
        else:
            # List of two tuples
            prev_qa = [[q for q, a in prev_qa], [a for q, a in prev_qa]]

            body = {
                "question": question,
                "context": context,
                "prev_q": prev_qa[0],
                "prev_a": prev_qa[1],
                "model": self.model
            }

            res = requests.post("https://api.kiri.ai/qa", json=body,
                                headers={"x-api-key": self.api_key}).json()

            return res["answer"]