from typing import List, Tuple
import pytorch_lightning as pl
from transformers.optimization import Adafactor
import torch
from torch.utils.data import DataLoader
from random import shuffle

from kiri.models import TextGenerationModel

class T5(TextGenerationModel, pl.LightningModule):
    def __init__(self, *args, model_path="t5-small", **kwargs):
        pl.LightningModule.__init__(self)

        TextGenerationModel.__init__(self, model_path,
                                *args, **kwargs)

        self.tasks = ["text-generation"]
        self.description = "This is the T5 model by Google."
        self.name = "t5"
        self.batch_size = 1
        self.hparams.batch_size = 1

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
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return Adafactor(params=self.model.parameters(), lr=1e-3, scale_parameter=False, relative_step=False)

    def encode(self, row):
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
    
    def train_dataloader(self):
        return DataLoader(self.dataset_train,
            batch_size=self.batch_size|self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid,
            batch_size=self.batch_size|self.hparams.batch_size)
    
    def finetune(self, input_text, output_text, validation_split=0.1):
        self.check_init()
        if not torch.cuda.is_available():
            raise Exception("You need a cuda capable (Nvidia) GPU for finetuning")

        assert len(input_text) == len(output_text)
        OPTIMAL_BATCH_SIZE = 128

        print("Processing data...")
        dataset = zip(input_text, output_text)
        dataset = list(map(self.encode, dataset))
        
        shuffle(dataset)

        dataset_train = dataset[:int(len(dataset) * (1 - validation_split))]
        dataset_valid = dataset[int(len(dataset) * (1 - validation_split)):]

        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        
        # Find batch size
        trainer = pl.Trainer(auto_scale_batch_size="power", gpus=-1)
        print("Finding the optimal batch size...")
        trainer.tune(self)

        batch_size = self.batch_size|self.hparams.batch_size

        # Don't go over
        batch_size = min(batch_size, OPTIMAL_BATCH_SIZE)

        accumulate_grad_batches = max(1, int(OPTIMAL_BATCH_SIZE / self.batch_size))

        trainer = pl.Trainer(gpus=-1, accumulate_grad_batches=int(OPTIMAL_BATCH_SIZE / batch_size),
            max_epochs=3, checkpoint_callback=False, logger=False)

        print("Starting to train...")
        self.model.train()
        trainer.fit(self)

        del self.dataset_train
        del self.dataset_valid

        print("Training finished! Save your model for later with kiri.save or upload it with kiri.upload")
