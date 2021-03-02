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
    def __init__(self, model: Union[str, BaseModel] = None,
                local: bool = False, api_key: str = None, device: str = "cpu",
                init: bool = False):

        super().__init__(model, local=local, api_key=api_key, device=device,
                        init=init, local_models=LOCAL_MODELS, api_models=API_MODELS,
                        default_local_model=DEFAULT_LOCAL_MODEL,
                        default_api_model=DEFAULT_API_MODEL)
    
    def __call__(self, question: str, context: str, prev_qa: List[Tuple[str, str]] = []):
        """
        Calls the qa model with question, context and list of previous question, answer pairs.
        """
        # List of two tuples
        prev_qa = [[q for q, a in prev_qa], [a for q, a in prev_qa]]

        task_input = {
            "question": question,
            "context": context,
            "prev_q": prev_qa[0],
            "prev_a": prev_qa[1],
        }
        if self.local:
            return self.model(task_input, task="qa")
        else:
            task_input["model"] = self.model

            res = requests.post("https://api.kiri.ai/qa", json=task_input,
                                headers={"x-api-key": self.api_key}).json()

            return res["answer"]