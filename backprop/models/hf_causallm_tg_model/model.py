import torch

from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from backprop.models import HFTextGenerationModel

class HFCausalLMTGModel(HFTextGenerationModel):
    def __init__(self, model_path=None, tokenizer_path=None, name: str = None,
                description: str = None, details: Dict = None, tasks: List[str] = None,
                model_class=AutoModelForCausalLM,
                tokenizer_class=AutoTokenizer, device=None):

        tasks = tasks or ["text-generation"]

        HFTextGenerationModel.__init__(self, model_path, name=name, description=description,
                    tasks=tasks, details=details, tokenizer_path=tokenizer_path,
                    model_class=model_class, tokenizer_class=tokenizer_class,
                    device=device)

    @staticmethod
    def list_models():
        from .models_list import models

        return models

    def __call__(self, task_input, task="text-generation"):
        """
        Uses the model for the text-generation task

        Args:
            task_input: input dictionary according to the ``text-generation`` task specification.
            task: text-generation
        """
        if task in ["text-generation", "generation"]:
            text = task_input.pop("text")
            temperature = task_input.pop("temperature", 0.7)

            # TODO: Set pad token on a model basis
            return self.generate(text, **task_input, pad_token_id=50256, temperature=temperature)
        else:
            raise ValueError(f"Unsupported task: {task}")