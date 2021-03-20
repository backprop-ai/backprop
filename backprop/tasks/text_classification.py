from typing import List, Tuple, Union, Optional
from backprop.models import BartLargeMNLI, XLMRLargeXNLI, BaseModel
from .base import Task

import requests

DEFAULT_LOCAL_MODEL = BartLargeMNLI

LOCAL_MODELS = {
    "english": DEFAULT_LOCAL_MODEL,
    "multilingual": XLMRLargeXNLI
}

DEFAULT_API_MODEL = "english"

FINETUNABLE_MODELS = ["xlnet"]

API_MODELS = ["english", "multilingual"]

class TextClassification(Task):
    """
    Task for classification.

    Attributes:
        model:
            1. Name of the model on Backprop's classification endpoint (english, multilingual or your own uploaded model)
            2. Officially supported local models (english, multilingual).
            3. Model class of instance Backprop's TextClassificationModel
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

    
    def __call__(self, text: Union[str, List[str]], labels: Optional[Union[List[str], List[List[str]]]] = None):
        """Classify input text based on previous training (user-tuned models) or according to given list of labels (zero-shot)

        Args:
            text: string or list of strings to be classified
            labels: list of labels for zero-shot classification (on our out-of-the-box models).
                    If using a user-trained model (e.g. XLNet), this is not used.

        Returns:
            dict where each key is a label and value is probability between 0 and 1, or list of dicts.
        """
        if self.local:
            task = "text-classification"
            
            task_input = {
                "text": text,
                "labels": labels
            }

            return self.model(task_input, task=task)
        else:
            body = {
                "text": text,
                "labels": labels,
                "model": self.model,
            }

            res = requests.post("https://api.backprop.co/text-classification", json=body,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            return res["probabilities"]
    
    def finetune(self, *args, **kwargs):
        """
        Passes the args and kwargs to the model's finetune method.
        """
        try:
            return self.model.finetune(*args, **kwargs)
        except NotImplementedError:
            raise NotImplementedError(f"This model does not support finetuning, try {', '.join(FINETUNABLE_MODELS)}")