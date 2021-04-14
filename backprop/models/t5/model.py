from typing import List, Tuple, Dict
import torch

from backprop.models import HFSeq2SeqTGModel

class T5(HFSeq2SeqTGModel):
    """
    Initialises a T5 model.

    Attributes:
        model_path: path to an appropriate T5 model on huggingface (t5-small)
        name: string identifier for the model. Lowercase letters and numbers.
            No spaces/special characters except dashes.
        description: String description of the model.
        tasks: List of supported task strings
        details: Dictionary of additional details about the model
        device: Device for model. Defaults to "cuda" if available.
    """
    def __init__(self, model_path=None, name: str = None,
                description: str = None, details: Dict = None, tasks: List[str] = None,
                device=None):
        tasks = tasks or ["text-generation", "summarisation"]
        HFSeq2SeqTGModel.__init__(self, model_path=model_path, name=name,
                                description=description, tasks=tasks, details=details,
                                device=device)

    @torch.no_grad()
    def __call__(self, task_input, task="text-generation"):
        """
        Uses the model for the chosen task

        Args:
            task_input: input dictionary according to the chosen task's specification
            task: one of text-generation, emotion, summarisation, qa 
        """
        if task == "text-generation":
            text = task_input.pop("text")
            return self.generate(text, **task_input)
        elif task == "summarisation":
            return self.summary(task_input["text"], "summarize",
                    no_repeat_ngram_size=3, num_beams=4, early_stopping=True)
        else:
            raise ValueError(f"Unsupported task: {task}")

    @staticmethod
    def list_models():
        from .models_list import models

        return models

    def summary(self, text, task_prefix, **gen_kwargs):
        if isinstance(text, list):
            text = [f"{task_prefix}: {t}" for t in text]
        else:
            text = f"{task_prefix}: {text}"
        
        return self.generate(text, do_sample=False, max_length=200, min_length=30, **gen_kwargs)