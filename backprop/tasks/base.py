from typing import Dict, List, Union, Tuple
import logging
from backprop import load
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import os
import torch
from torch.utils.data import Subset, Dataset
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from pytorch_lightning.callbacks import EarlyStopping

logger = logging.getLogger("info")

class Task(pl.LightningModule):
    def __init__(self, model, local=False, api_key=None,
                device: str = None, local_models: Dict = None,
                api_models: List[str] = None, default_local_model: str = None,
                default_api_model: str = None):
        super().__init__()

        if api_key == None:
            local = True
                    
        if device == None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.local = local
        self.api_key = api_key
        self._model_device = device

        # Pick the correct model name
        if local:
            if model is None:
                model = default_local_model

            if type(model) == str or model is None:
                # Get from dictionary or use provided if not there
                model = local_models.get(model) or model

            if type(model) == str:
                self.model = load(model)
            elif hasattr(model, "model"):
                self.model = model
            else:
                self.model = model(device=device)
        else:
            if model is None or type(model) != str:
                model = default_api_model
    
            # All checks passed
            self.model = model

    def __call__(self):
        raise Exception("The base Task is not callable!")

    def configure_optimizers(self):
        raise NotImplementedError("configure_optimizers is not implemented for this task")

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def step(self, batch, batch_idx):
        raise NotImplementedError("step is not implemented for this task")

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() or 0)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() or 0)

    def finetune(self, dataset = None, validation_split: Union[float, Tuple[List[int], List[int]]] = 0.15,
                epochs: int = 20, batch_size: int = None, optimal_batch_size: int = None,
                early_stopping_epochs: int = 1, train_dataloader = None, val_dataloader = None,
                dataset_train: Dataset = None, dataset_valid: Dataset = None, step = None):
        self.batch_size = batch_size or 1

        if not torch.cuda.is_available():
            raise Exception("You need a cuda capable (Nvidia) GPU for finetuning")

        if dataset == None:
            if not dataset_train and not dataset_valid:
                raise ValueError("Must either provide dataset or dataset_train and dataset_valid")

        # Override dataloaders if provided
        if train_dataloader:
            self.train_dataloader = train_dataloader

        if val_dataloader:
            self.val_dataloader = val_dataloader

        # Make training and validation datasets
        if not dataset_train or not dataset_valid:
            if isinstance(validation_split, tuple):
                train_idx, val_idx = validation_split

                dataset_train = Subset(dataset, train_idx)
                dataset_valid = Subset(dataset, val_idx)
            elif type(validation_split) == float:
                len_train = int(len(dataset) * (1 - validation_split))
                len_valid = len(dataset) - len_train
                dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [len_train, len_valid])
            else:
                raise ValueError(f"Unsupported type '{type(validation_split)}' for validation_split")

        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid

        # Override step if provided
        if step:
            self.step = step

        # Find batch size automatically
        if batch_size == None:
            temp_trainer = pl.Trainer(auto_scale_batch_size="power", gpus=-1)
            print("Finding the optimal batch size...")
            temp_trainer.tune(self)

            # Ensure that memory gets cleared
            del self.trainer
            del temp_trainer
            garbage_collection_cuda()

        trainer_kwargs = {}
        
        # Accumulate grad to optimal batch size
        if optimal_batch_size:
            # Don't go over limit
            batch_size = min(self.batch_size, optimal_batch_size)
            accumulate_grad_batches = max(1, int(optimal_batch_size / batch_size))
            trainer_kwargs["accumulate_grad_batches"] = accumulate_grad_batches

        # Stop when val loss doesn't improve after a number of epochs
        if early_stopping_epochs != 0:
            early_stopping = EarlyStopping(monitor="val_loss", patience=early_stopping_epochs)
            trainer_kwargs["callbacks"] = [early_stopping]

        trainer = pl.Trainer(gpus=-1, max_epochs=epochs, checkpoint_callback=False,
                            logger=False, **trainer_kwargs)

        self.model.train()
        trainer.fit(self)

        # For some reason the model can end up on CPU after training
        self.model.to(self.model._model_device)
        self.model.eval()
        print("Training finished! Save your model for later with backprop.save or upload it with backprop.upload")