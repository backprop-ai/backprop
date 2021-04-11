from typing import Dict, List, Union, Tuple
import logging
import backprop
from backprop.models import AutoModel
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import os
import torch
from torch.utils.data import Subset, Dataset
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from pytorch_lightning.callbacks import EarlyStopping

logger = logging.getLogger("info")

class Task(pl.LightningModule):
    """
    Base Task superclass used to implement new tasks.

    Attributes:
        model: Model name string for the task in use.
        local: Run locally. Defaults to False.
        api_key: Backprop API key for non-local inference.
        device: Device to run inference on. Defaults to "cuda" if available.
        models: All supported models for a given task (pulls from config).
        default_local_model: Which model the task will default to if initialized with none provided: Defined per-task.
    """

    def __init__(self, model, local=False, api_key=None,
                task: str = None,
                device: str = None, models: Dict = None,
                default_local_model: str = None,
                local_aliases: Dict = None):
        super().__init__()

        if api_key == None:
            local = True
        
        self.task = task
        self.local = local
        self.api_key = api_key

        # Pick the correct model name
        if local:
            if model is None:
                model = default_local_model

            if type(model) == str:
                model = AutoModel.from_pretrained(model, aliases=local_aliases, device=device)

            if task not in model.tasks:
                raise ValueError(f"Model does not support the '{task}' task")

        else:
            if model is not None and type(model) != str:
                raise ValueError(f"Model must be a string identifier to be used in the API")
    
            # API uses default model
            if model is None:
                model = ""

        # All checks passed
        self.model = model

    def __call__(self):
        raise Exception("The base Task is not callable!")

    def save(self, name: str, description: str = None, details: Dict = None):
        """
        Saves the model used by task to ``~/.cache/backprop/name``

        Args:
            name: string identifier for the model. Lowercase letters and numbers.
                No spaces/special characters except dashes.
            description: String description of the model.
            details: Valid json dictionary of additional details about the model
        """
        return backprop.save(self.model, name=name, description=description, details=details)

    def upload(self, name: str, description: str = None, details: Dict = None, api_key: str = None):
        """
        Saves the model used by task to ``~/.cache/backprop/name`` and deploys to backprop

        Args:
            name: string identifier for the model. Lowercase letters and numbers.
                No spaces/special characters except dashes.
            description: String description of the model.
            details: Valid json dictionary of additional details about the model
            api_key: Backprop API key
        """
        return backprop.upload(self.model, name=name, description=description, details=details, api_key=api_key)

    def configure_optimizers(self):
        """
        Sets up optimizers for model. Must be defined in task: no base default.
        """
        raise NotImplementedError("configure_optimizers is not implemented for this task")

    def training_step(self, batch, batch_idx):
        """
        Performs the step function with training data and gets training loss.

        Args:
            batch: Batch output from dataloader.
            batch_idx: Batch index.
        """
        loss = self.step(batch, batch_idx)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Performs the step function with validation data and gets validation loss.

        Args:
            batch: Batch output from dataloader.
            batch_idx: Batch index.
        """
        loss = self.step(batch, batch_idx)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def step(self, batch, batch_idx):
        """
        Implemented per-task, passes batch into model and returns loss.
        
        Args:
            batch: Batch output from dataloader.
            batch_idx: Batch index.
        """
        raise NotImplementedError("step is not implemented for this task")

    def train_dataloader(self):
        """
        Returns a default dataloader of training data.
        """
        return DataLoader(self.dataset_train,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() or 0)

    def val_dataloader(self):
        """
        Returns a default dataloader of validation data.
        """
        return DataLoader(self.dataset_valid,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() or 0)

    def finetune(self, dataset = None, validation_split: Union[float, Tuple[List[int], List[int]]] = 0.15,
                epochs: int = 20, batch_size: int = None, optimal_batch_size: int = None,
                early_stopping_epochs: int = 1, train_dataloader = None, val_dataloader = None,
                dataset_train: Dataset = None, dataset_valid: Dataset = None, step = None, configure_optimizers = None):
        self.batch_size = batch_size or 1
        """
        Core finetuning logic followed for all implemented tasks.

        Args:
            dataset: Torch dataset on which training will occur
            validation_split: Float between 0 and 1 that determines percentage of data to use for validation
            epochs: Integer specifying how many training iterations to run
            batch_size: Batch size when training. Leave as None to automatically determine batch size.
            optimal_batch_size: 
            early_stopping_epochs: Integer determining how many epochs will run before stopping without an improvement in validation loss
            train_dataloader: Dataloader for providing training data when finetuning. Defaults to inbuilt dataloder.
            val_dataloader: Dataloader for providing validation data when finetuning. Defaults to inbuilt dataloader.
            dataset_train: Torch dataset of training data (split before provided). Automatically made from dataset if None.
            dataset_valid: Torch dataset of validation data (split before provided). Automatically made from dataset if None.
            step: Function determining how to call model for a training step. Implemented per-task, can be overridden.
            configure_optimizers: Function that sets up the optimizer for training. Implemented per-task, can be overridden.
        """

        model_task_details = self.model.details.get(self.task)

        if not model_task_details or not model_task_details.get("finetunable"):
            raise NotImplementedError(f"Finetuning has not been implemented for model '{self.model.name}' for task '{self.task}'")

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

        # Override optimizer if provided
        if configure_optimizers:
            self.configure_optimizers = configure_optimizers

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