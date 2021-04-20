import torch

from typing import List, Dict
import transformers
from backprop.models import HFTextGenerationModel

class HFCausalLMTGModel(HFTextGenerationModel):
    """
    Class for Hugging Face causal LM models

    Attributes:
        model_path: path to HF model
        tokenizer_path: path to HF tokenizer
        name: string identifier for the model. Lowercase letters and numbers.
            No spaces/special characters except dashes.
        description: String description of the model.
        tasks: List of supported task strings
        details: Dictionary of additional details about the model
        model_class: Class used to initialise model
        tokenizer_class: Class used to initialise tokenizer
        device: Device for model. Defaults to "cuda" if available.
    """
    def __init__(self, model_path=None, tokenizer_path=None, name: str = None,
                description: str = None, details: Dict = None, tasks: List[str] = None,
                model_class=transformers.AutoModelForCausalLM,
                tokenizer_class=transformers.AutoTokenizer, device=None):

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
        if task == "text-generation":
            text = task_input.pop("text")
            temperature = task_input.pop("temperature", 0.7)

            # TODO: Set pad token on a model basis
            return self.generate(text, **task_input, pad_token_id=50256, temperature=temperature, variant="causal_lm")
        else:
            raise ValueError(f"Unsupported task: {task}")