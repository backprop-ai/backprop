from typing import List, Tuple, Dict
from transformers import AutoModelForPreTraining, AutoTokenizer, \
    AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Subset
from sentence_transformers import SentenceTransformer
from functools import partial
import os

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.memory import garbage_collection_cuda

class BaseModel(torch.nn.Module):
    """
    The base class for a model.
    
    Attributes:
        model: Your model that takes some args, kwargs and returns an output.
            Must be callable.
        name: string identifier for the model. Lowercase letters and numbers.
            No spaces/special characters except dashes.
        description: String description of the model.
        tasks: List of supported task strings
        details: Dictionary of additional details about the model
    """
    def __init__(self, model, name: str = None, description: str = None, tasks: List[str] = None,
                details: Dict = None):
        torch.nn.Module.__init__(self)
        self.model = model
        self.name = name or "base-model"
        self.description = description or "This is the base description. Change me."
        self.tasks = tasks or [] # Supports no tasks
        self.details = details or {}

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def finetune(self, *args, **kwargs):
        raise NotImplementedError("This model does not support finetuning")

    def to(self, device):
        self.model.to(device)
        self._model_device = device
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class PathModel(BaseModel):
    """
    Class for models which are initialised from a path.

    Attributes:
        model_path: Path to the model
        name: string identifier for the model. Lowercase letters and numbers.
            No spaces/special characters except dashes.
        description: String description of the model.
        tasks: List of supported task strings
        details: Dictionary of additional details about the model
        init_model: Callable to initialise model from path
        tokenizer_path (optional): Path to the tokenizer
        init_tokenizer (optional): Callable to initialise tokenizer from path
        device (optional): Device for inference. Defaults to "cuda" if available.
    """
    def __init__(self, model_path, init_model, name: str = None,
                description: str = None, tasks: List[str] = None,
                details: Dict = None, tokenizer_path=None,
                init_tokenizer=None, device=None):
        BaseModel.__init__(self, None, name=name, description=description, tasks=tasks, details=details)
        self.init_model = init_model
        self.init_tokenizer = init_tokenizer
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self._model_device = device

        if self._model_device is None:
            self._model_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialise
        self.model = self.init_model(model_path).eval().to(self._model_device)

        # Not all models need tokenizers
        if self.tokenizer_path:
            self.tokenizer = self.init_tokenizer(self.tokenizer_path)
        

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class HFModel(PathModel):
    """
    Class for huggingface models

    Attributes:
        model_path: Local or huggingface.co path to the model
        name: string identifier for the model. Lowercase letters and numbers.
            No spaces/special characters except dashes.
        description: String description of the model.
        tasks: List of supported task strings
        details: Dictionary of additional details about the model
        init_model: Callable to initialise model from path
            Defaults to AutoModelForPreTraining from huggingface
        tokenizer_path (optional): Path to the tokenizer
        init_tokenizer (optional): Callable to initialise tokenizer from path
            Defaults to AutoTokenizer from huggingface.
        device (optional): Device for inference. Defaults to "cuda" if available.
    """
    def __init__(self, model_path, tokenizer_path=None, name: str = None,
                description: str = None, tasks: List[str] = None, details: Dict = None,
                model_class=AutoModelForPreTraining,
                tokenizer_class=AutoTokenizer, device=None):
        # Usually the same
        if not tokenizer_path:
            tokenizer_path = model_path

        # Object was made with init = False
        if hasattr(self, "initialised"):
            model_path = self.model_path
            tokenizer_path = self.tokenizer_path
            init_model = self.init_model
            init_tokenizer = self.init_tokenizer
            device = self._model_device
        else:
            init_model = model_class.from_pretrained
            init_tokenizer = tokenizer_class.from_pretrained

        return PathModel.__init__(self, model_path, name=name, description=description,
                                tasks=tasks,
                                details=details,
                                tokenizer_path=tokenizer_path,
                                init_model=init_model,
                                init_tokenizer=init_tokenizer,
                                device=device)


class HFTextGenerationModel(HFModel):
    """
    Class for huggingface models that implement the .generate method.

    Attributes:
        *args and **kwargs are passed to HFModel's __init__
    """
    def generate(self, text, variant="seq2seq", **kwargs):
        """
        Generate according to the model's generate method.
        """
        # Get and remove do_sample or set to False
        do_sample = kwargs.pop("do_sample", None) or False
        params = ["temperature", "top_k", "top_p", "repetition_penalty",
                    "length_penalty", "num_beams", "num_return_sequences", "num_generations"]

        # If params are changed, we want to sample
        for param in params:
            if param in kwargs.keys() and kwargs[param] != None:
                do_sample = True
                break

        if "temperature" in kwargs:
            # No sampling
            if kwargs["temperature"] == 0.0:
                do_sample = False
                del kwargs["temperature"]

        # Override, name correctly
        if "num_generations" in kwargs:
            if kwargs["num_generations"] != None:
                kwargs["num_return_sequences"] = kwargs["num_generations"]
                
            del kwargs["num_generations"]

        min_length = kwargs.pop("min_length", 10)
        max_length = kwargs.pop("max_length", 20)

        is_list = False
        if isinstance(text, list):
            is_list = True

        if not is_list:
            text = [text]

        all_tokens = []
        output = []
        for text in text:
            features = self.tokenizer(text, return_tensors="pt")

            input_length = len(features["input_ids"][0])

            for k, v in features.items():
                features[k] = v.to(self._model_device)

            if variant == "causal_lm":
                min_length += input_length
                max_length += input_length

            with torch.no_grad():
                tokens = self.model.generate(do_sample=do_sample,
                                            min_length=min_length,
                                            max_length=max_length,
                                            **features, **kwargs)
            
            if variant == "causal_lm":
                output.append([self.tokenizer.decode(tokens[input_length:], skip_special_tokens=True)
                        for tokens in tokens])
            else:
                output.append([self.tokenizer.decode(tokens, skip_special_tokens=True)
                        for tokens in tokens])

        # Unwrap generation list
        if kwargs.get("num_return_sequences", 1) == 1:
            output_unwrapped = []
            for value in output:
                output_unwrapped.append(value[0])

            output = output_unwrapped
        
        # Return single item
        if not is_list:
            output = output[0]

        return output