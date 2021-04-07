from typing import List, Tuple
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers.optimization import Adafactor
import torch
from torch.utils.data import DataLoader
from random import shuffle
import os

from backprop.models import TextGenerationModel, Finetunable

class T5(TextGenerationModel):
    """
    Google's T5 model for text-generation.

    Attributes:
        args: args passed to :class:`backprop.models.generic_models.TextGenerationModel`
        model_path: path to a T5 model on huggingface (t5-small, t5-base, t5-large)
        kwargs: kwargs passed to :class:`backprop.models.generic_models.TextGenerationModel`
    """
    def __init__(self, *args, model_path="t5-small", **kwargs):
        TextGenerationModel.__init__(self, model_path,
                                *args, **kwargs)

        self.tasks = ["text-generation"]
        self.description = "This is the T5 model by Google."
        self.name = "t5"
        self.optimal_batch_size = 128

    def __call__(self, task_input, task="text-generation", train=False):
        """
        Uses the model for the text-generation task

        Args:
            task_input: input dictionary according to the ``text-generation`` task specification,
                        or specification of task you've finetuned for.
            task: text-generation, or a task you've tuned a T5 model on
        """
        if task == "text-generation":
            with torch.set_grad_enabled(train):
                if train:
                    return self.model(**task_input)
                else:
                    text = task_input.pop("text")
                    return self.generate(text, **task_input)
        else:
            raise ValueError(f"Unsupported task: {task}")

    def process_text(self, input_text, output_text=None, max_input_length=128,
                    max_output_length=32):
        inp = self.encode_input(input_text, max_length=max_input_length)
        processed = {**inp}

        if output_text:
            out = self.encode_output(output_text, max_length=max_output_length)
            processed = {**processed, **out}

        return processed
    
    def encode_input(self, text, max_length=128):
        tokens = self.tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        return {"input_ids": tokens.input_ids[0], "attention_mask": tokens.attention_mask[0]}

    def encode_output(self, text, max_length=32):
        tokens =  self.tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        return {"labels": tokens.input_ids[0], "decoder_attention_mask": tokens.attention_mask[0]}