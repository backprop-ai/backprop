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
            1. Name of the model on Kiri's qa endpoint (english or your own uploaded model)
            2. Officially supported local models (english).
            3. Model class of instance Kiri's BaseModel that implements the qa task
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
    
    def __call__(self, question: Union[str, List[str]], context: Union[str, List[str]],
                prev_qa: Union[List[Tuple[str, str]], List[List[Tuple[str, str]]]] = []):
        """Perform QA, either on docstore or on provided context.

        Args:
            question: Question (string or list of strings) for qa model.
            context: Context (string or list of strings) to ask question from.
            prev_qa (optional): List of previous question, answer tuples or list of prev_qa.

        Returns:
            Answer string or list of answer strings
        """

        prev_q = []
        prev_a = []
        if prev_qa != [] and type(prev_qa[0]) == list:
            for prev_qa in prev_qa:
                prev_q += [q for q, a in prev_qa]
                prev_a += [a for q, a in prev_qa]
        else:
            prev_q = [q for q, a in prev_qa]
            prev_a = [a for q, a in prev_qa]

        task_input = {
            "question": question,
            "context": context,
            "prev_q": prev_q,
            "prev_a": prev_q,
        }
        if self.local:
            return self.model(task_input, task="qa")
        else:
            task_input["model"] = self.model

            res = requests.post("https://api.kiri.ai/qa", json=task_input,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            return res["answer"]