from typing import Dict, List

import torch
from functools import partial
from backprop.models import PathModel
from torch.optim.adamw import AdamW
from sentence_transformers import SentenceTransformer

class STModel(PathModel):
    """
    Class for models which are initialised from Sentence Transformers

    Attributes:
        model_path: path to ST model
        name: string identifier for the model. Lowercase letters and numbers.
            No spaces/special characters except dashes.
        max_length: Max supported token length for vectorisation
        description: String description of the model.
        tasks: List of supported task strings
        details: Dictionary of additional details about the model
        init_model: Class used to initialise model
        device: Device for model. Defaults to "cuda" if available.
    """
    def __init__(self, model_path, init_model=SentenceTransformer, name: str = None,
                description: str = None, tasks: List[str] = None, details: Dict = None,
                max_length=512, device=None):
        init_model = partial(init_model, device=device)

        tasks = ["text-vectorisation"]

        PathModel.__init__(self, model_path, name=name, description=description,
                                details=details, tasks=tasks,
                                init_model=init_model,
                                device=device)

        self.max_length = max_length

    @staticmethod
    def list_models():
        from .models_list import models

        return models

    @torch.no_grad()
    def __call__(self, task_input, task="text-vectorisation", return_tensor=False):
        """
        Uses the model for the text-vectorisation task

        Args:
            task_input: input dictionary according to the ``text-vectorisation`` task specification
            task: text-vectorisation
        """
        is_list = False
        if task == "text-vectorisation":
            input_ids = None
            attention_mask = None            
            
            text = task_input.get("text")

            if type(text) == list:
                is_list = True
            else:
                text = [text]

            features = self.model.tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(self._model_device)

            text_vecs = self.vectorise(features)

            if not return_tensor:
                text_vecs = text_vecs.tolist()

            output = text_vecs

            if not is_list:
                output = output[0]

            return output
        else:
            raise ValueError(f"Unsupported task '{task}'")
    
    def training_step(self, params, task="text-vectorisation"):
        text = params["text"]
        return self.vectorise(text)

    def process_batch(self, params, task="text-vectorisation"):
        if task == "text-vectorisation":
            max_length = params["max_length"] or self.max_length
            if max_length > self.max_length:
                raise ValueError(f"This model has a max_length limit of {self.max_length}")
            text = params["text"]
            return self.model.tokenizer(text, truncation=True, padding="max_length", return_tensors="pt")
                
    def vectorise(self, features):
        return self.model.forward(features)["sentence_embedding"]

    def configure_optimizers(self):
        return AdamW(params=self.model.parameters(), lr=2e-5, eps=1e-6, correct_bias=False)