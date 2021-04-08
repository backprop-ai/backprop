from typing import Dict, List

import torch
from functools import partial
from backprop.models import PathModel
from torch.optim.adamw import AdamW
from sentence_transformers import SentenceTransformer

class STModel(PathModel):
    """
    Class for models which are initialised from a local path or Sentence Transformers

    Attributes:
        model_path: Local, sentence transformers or huggingface.co path to the model
        init_model: Callable to initialise model from path
            Defaults to SentenceTransformer
        device (optional): Device for inference. Defaults to "cuda" if available.
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

    def __call__(self, task_input, task="text-vectorisation", return_tensor=False, preprocess=True, train=False):
        is_list = False
        if task == "text-vectorisation":
            input_ids = None
            attention_mask = None

            if preprocess:
                text = task_input.get("text")

                if type(text) == list:
                    is_list = True
                else:
                    text = [text]

                features = self.process_text(text).to(self._model_device)
            else:
                features = task_input.get("text")
                is_list = True

            with torch.set_grad_enabled(train):
                text_vecs = self.vectorise(features)

            if not return_tensor:
                text_vecs = text_vecs.tolist()

            output = text_vecs

            if not is_list:
                output = output[0]

            return output
        else:
            raise ValueError(f"Unsupported task '{task}'")

    def process_text(self, input_text, max_length=None, padding=True):
        max_length = max_length or self.max_length

        if max_length > self.max_length:
            raise ValueError(f"This model has a max_length limit of {self.max_length}")

        processed = self.model.tokenizer(input_text, truncation=True, padding=padding,
                    return_tensors="pt", max_length=max_length)

        return processed

    def vectorise(self, features):
        return self.model.forward(features)["sentence_embedding"]

    def configure_optimizers(self):
        return AdamW(params=self.model.parameters(), lr=2e-5, eps=1e-6, correct_bias=False)