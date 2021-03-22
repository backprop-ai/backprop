from typing import List, Tuple, Union, Dict
from .base import Task
from backprop.models import T5QASummaryEmotion, BaseModel

import requests

DEFAULT_LOCAL_MODEL = T5QASummaryEmotion

LOCAL_MODELS = {
    "english": DEFAULT_LOCAL_MODEL
}

DEFAULT_API_MODEL = "english"

FINETUNABLE_MODELS = ["t5", "t5-base-qa-summary-emotion"]

API_MODELS = ["english"]

class Emotion(Task):
    """
    Task for emotion detection.

    Attributes:
        model:
            1. Name of the model on Backprop's emotion endpoint (english or your own uploaded model)
            2. Officially supported local models (english).
            3. Model class of instance Backprop's BaseModel that implements the emotion task
            4. Path/name of saved Backprop model
        model_class (optional): The model class to use when supplying a path for the model.
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

            res = requests.post("https://api.backprop.co/emotion", json=task_input,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            return res["emotion"]
    
    def finetune(self, params: Dict, *args, **kwargs):
        """
        Passes args and kwargs to the model's finetune method.

        Args:
            params: dictionary of 'input_text' and 'output_text' lists.
        """

        if not "input_text" in params:
            print("Params requires key: 'input_text' (list of inputs)")
            return
        if not "output_text" in params:
            print("Params requires key: 'output_text' (list of outputs)")
            return

        try:
            return self.model.finetune(params, task="emotion", *args, **kwargs)
        except NotImplementedError:
            raise NotImplementedError(f"This model does not support finetuning, try: {', '.join(FINETUNABLE_MODELS)}")
    
