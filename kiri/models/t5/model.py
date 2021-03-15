from typing import List, Tuple
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers.optimization import Adafactor
import torch
from torch.utils.data import DataLoader
from random import shuffle
import os

from kiri.models import TextGenerationModel, Finetunable

class T5(TextGenerationModel, Finetunable):
    def __init__(self, *args, model_path="t5-small", **kwargs):
        Finetunable.__init__(self)
        TextGenerationModel.__init__(self, model_path,
                                *args, **kwargs)

        self.batch_size = 1
        self.tasks = ["text-generation", "generation"]
        self.description = "This is the T5 model by Google."
        self.name = "t5"

    def __call__(self, task_input, task="text-generation"):
        if task in ["text-generation", "generation"]:
            text = task_input.pop("text")

            return self.generate(text, **task_input)
        else:
            raise ValueError(f"Unsupported task: {task}")
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return Adafactor(params=self.model.parameters(), lr=1e-3, scale_parameter=False, relative_step=False)

    def encode(self, row, max_input_length, max_output_length):
        inp = self.encode_input(row[0])
        out = self.encode_output(row[1])

        row = {**inp, **out}
        return row
    
    def encode_input(self, text, max_length=128):
        tokens = self.tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        return {"input_ids": tokens.input_ids[0], "attention_mask": tokens.attention_mask[0]}

    def encode_output(self, text, max_length=32):
        tokens =  self.tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        return {"labels": tokens.input_ids[0], "decoder_attention_mask": tokens.attention_mask[0]}
    
    def finetune(self, input_text: List[str], output_text: List[str],
                max_input_length=128, max_output_length=32,
                validation_split: float = 0.15, epochs: int = 20):
        """
        Finetunes T5 for a text generation task.
        input_text and output_text must be ordered the same way (item 1 of input must match item 1 of output)

        Args:
            input_text: List of strings that are used to predict and output (must match output ordering)
            output_text: List of strings that are predicted using input (must match input ordering)
            validation_split: Float between 0 and 1 that determines what percentage of the data to use for validation
            epochs: Integer that specifies how many iterations of training to do
        """
        self.check_init()
        if not torch.cuda.is_available():
            raise Exception("You need a cuda capable (Nvidia) GPU for finetuning")

        assert len(input_text) == len(output_text)
        OPTIMAL_BATCH_SIZE = 128

        print("Processing data...")
        dataset = zip(input_text, output_text)
        dataset = [self.encode(r, max_input_length, max_output_length) for r in dataset]
        
        Finetunable.finetune(self, dataset, validation_split=validation_split,
            epochs=epochs, optimal_batch_size=OPTIMAL_BATCH_SIZE)