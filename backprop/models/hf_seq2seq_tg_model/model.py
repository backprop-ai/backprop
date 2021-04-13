import torch
from typing import List, Dict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from backprop.models import HFTextGenerationModel

class HFSeq2SeqTGModel(HFTextGenerationModel):
    """
    Class for Hugging Face causal Seq2Seq generation models.

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
                model_class=AutoModelForSeq2SeqLM,
                tokenizer_class=AutoTokenizer, device=None):
        tasks = tasks or ["text-generation"]
        
        HFTextGenerationModel.__init__(self, model_path, name=name, description=description,
                    tasks=tasks, details=details, tokenizer_path=tokenizer_path,
                    model_class=model_class, tokenizer_class=tokenizer_class,
                    device=device)


    @torch.no_grad()
    def __call__(self, task_input, task="text-generation"):
        """
        Uses the model for the text-generation task

        Args:
            task_input: input dictionary according to the ``text-generation`` task specification
            task: text-generation
        """
        if task == "text-generation":
            text = task_input.pop("text")
            return self.generate(text, **task_input, variant="seq2seq")
        else:
            raise ValueError(f"Unsupported task: {task}")

    @staticmethod
    def list_models():
        from .models_list import models

        return models

    def training_step(self, task_input):
        return self.model(**task_input).loss

    def process_batch(self, params, task):
        inp = params["input"]
        out = params.pop("output", None)

        inp = self.encode_input(inp, max_length=params["max_input_length"])

        processed = {**inp}

        if out:
            out = self.encode_output(out, max_length=params["max_output_length"])
            processed = {**inp, **out}
        
        return processed
        
    def encode_input(self, text, max_length=128):
        tokens = self.tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        return {"input_ids": tokens.input_ids[0], "attention_mask": tokens.attention_mask[0]}

    def encode_output(self, text, max_length=32):
        tokens =  self.tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        return {"labels": tokens.input_ids[0], "decoder_attention_mask": tokens.attention_mask[0]}