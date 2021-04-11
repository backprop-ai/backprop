from typing import List, Tuple, Union
from .base import Task
from backprop.models import AutoModel, BaseModel

import requests
import torch.nn.functional as F
import torch
from torch.utils.data.dataloader import DataLoader
import os
import random
from functools import partial
from backprop.utils.datasets import TextGroupDataset, TextPairDataset
from backprop.utils.samplers import SameGroupSampler
from backprop.utils.losses.triplet_loss import TripletLoss

TASK = "text-vectorisation"

DEFAULT_LOCAL_MODEL = "msmarco-distilroberta-base-v2"

LOCAL_ALIASES = {
    "clip": "clip-vit-b32"
}

class TextVectorisation(Task):
    """
    Task for text vectorisation.

    Attributes:
        model:
            1. Model name
            2. Model name on Backprop's text-vectorisation endpoint
            3. Model object that implements the text-vectorisation task
        local (optional): Run locally. Defaults to False
        api_key (optional): Backprop API key for non-local inference
        device (optional): Device to run inference on. Defaults to "cuda" if available.
    """
    def __init__(self, model: Union[str, BaseModel] = None,
                local: bool = False, api_key: str = None, device: str = None):
        models = AutoModel.list_models(task=TASK)

        super().__init__(model, local=local, api_key=api_key, device=device,
                        models=models, task=TASK,
                        default_local_model=DEFAULT_LOCAL_MODEL,
                        local_aliases=LOCAL_ALIASES)

    @staticmethod
    def list_models(return_dict=False, display=False, limit=None):
        """
        Returns the list of models that can be used and finetuned with this task.

        Args:
            return_dict: Default False. True if you want to return in dict form. Otherwise returns list form.
            display: Default False. True if you want output printed directly (overrides return_dict, and returns nothing).
            limit: Default None. Maximum number of models to return -- leave None to get all models.
        """
        return AutoModel.list_models(task=TASK, return_dict=return_dict, display=display, limit=limit, aliases=LOCAL_ALIASES)
    
    def __call__(self, text: Union[str, List[str]], return_tensor=False):
        """Vectorise input text.

        Args:
            text: string or list of strings to vectorise. Can be both PIL Image objects or paths to images.

        Returns:
            Vector or list of vectors
        """
        vector = None

        if self.local:
            task_input = {
                "text": text
            }
            vector = self.model(task_input, task=TASK)
        else:
            body = {
                "text": text,
                "model": self.model
            }

            res = requests.post("https://api.backprop.co/text-vectorisation", json=body,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            vector = res["vector"]
        
        if return_tensor and not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector)

        return vector

    def configure_optimizers(self):
        """
        Returns default optimizer for text vectorisation (AdamW, learning rate 1e-5)
        """
        return torch.optim.AdamW(params=self.model.parameters(), lr=1e-5)

    def step_triplet(self, batch, batch_idx):
        """
        Performs a training step and calculates triplet loss.
        
        Args:
            batch: Batch output from dataloader.
            batch_idx: Batch index.
        """
        text, group = batch

        text_vecs_norm = self.model.training_step({"text": text}, task=TASK)

        loss = self.criterion(text_vecs_norm, group)

        return loss

    def step_cosine(self, batch, batch_idx):
        """
        Performs a training step and calculates cosine similarity loss.
        
        Args:
            batch: Batch output from dataloader.
            batch_idx: Batch index.
        """
        texts1, texts2, similarity_scores = batch

        text_vecs1_norm = self.model.training_step({"text": texts1}, task=TASK)
        text_vecs2_norm = self.model.training_step({"text": texts2}, task=TASK)

        loss = torch.cosine_similarity(text_vecs1_norm, text_vecs2_norm)
        loss = F.mse_loss(loss, similarity_scores.view(-1))

        return loss

    def train_dataloader_triplet(self):
        """
        Returns training dataloader with triplet loss sampling strategy.
        """
        return DataLoader(self.dataset_train,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() or 0,
            sampler=self.dl_sampler(self.dataset_train))

    def val_dataloader_triplet(self):
        """
        Returns validation dataloader with triplet loss sampling strategy.
        """
        return DataLoader(self.dataset_valid,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() or 0,
            sampler=self.dl_sampler(self.dataset_valid))

    def finetune(self, params, validation_split: Union[float, Tuple[List[int], List[int]]] = 0.15,
                max_length: int = None, variant: str = "cosine_similarity", epochs: int = 20, batch_size: int = None,
                optimal_batch_size: int = None, early_stopping_epochs: int = 1,
                train_dataloader = None, val_dataloader = None, step = None, configure_optimizers = None):
        """
        Finetunes a model for text vectorisation. Includes different variants for calculating loss.

        Args:
            params: Dictionary of model inputs.
                    If using triplet variant, contains keys "texts" and "groups".
                    If using cosine_similarity variant, contains keys "texts1", "texts2", and "similarity_scores".
            validation_split: Float between 0 and 1 that determines percentage of data to use for validation.
            max_length: Int determining the maximum token length of input strings.
            variant: How loss will be calculated: "cosine_similarity" (default) or "triplet".
            epochs: Integer specifying how many training iterations to run.
            batch_size: Batch size when training. Leave as None to automatically determine batch size.
            optimal_batch_size: Optimal batch size for the model being trained -- defaults to model settings.
            early_stopping_epochs: Integer determining how many epochs will run before stopping without an improvement in validation loss.
            train_dataloader: Dataloader for providing training data when finetuning. Defaults to inbuilt dataloder.
            val_dataloader: Dataloader for providing validation data when finetuning. Defaults to inbuilt dataloader.
            step: Function determining how to call model for a training step. Defaults to step defined in this task class.
            configure_optimizers: Function that sets up the optimizer for training. Defaults to optimizer defined in this task class.
        
        Examples::

            import backprop

            tv = backprop.TextVectorisation()

            # Set up training data & finetune (cosine_similarity variant)
            texts1 = ["I went to the store and bought some bread", "I am getting a cat soon"]
            texts2 = ["I bought bread from the store", "I took my dog for a walk"]
            similarity_scores = [1.0, 0.0]
            params = {"texts1": texts1, "texts2": texts2, "similarity_scores": similarity_scores}

            tv.finetune(params, variant="cosine_similarity")

            # Set up training data & finetune (triplet variant)
            texts = ["I went to the store and bought some bread", "I bought bread from the store", "I'm going to go walk my dog"]
            groups = [0, 0, 1]
            params = {"texts": texts, "groups": groups}

            tv.finetune(params, variant="triplet")
        """

        optimal_batch_size = getattr(self.model, "optimal_batch_size", 128)

        configure_optimizers = configure_optimizers or self.configure_optimizers

        process_batch = self.model.process_batch

        if variant == "triplet":
            texts = params["texts"]
            groups = params["groups"]
            assert len(texts) == len(groups), "The input lists must match"

            step = step or self.step_triplet

            if isinstance(validation_split, tuple):
                train_idx, val_idx = validation_split
            else:
                all_idx = list(range(len(texts)))
                val_len = int(len(all_idx) * validation_split)
                val_idx = random.sample(all_idx, val_len)
                train_idx = list(set(all_idx) - set(val_idx))
            
            dataset_train = TextGroupDataset(
                [texts[i] for i in train_idx],
                [groups[i] for i in train_idx],
                process_batch=process_batch,
                max_length=max_length
            )

            dataset_valid = TextGroupDataset(
                [texts[i] for i in val_idx],
                [groups[i] for i in val_idx],
                process_batch=process_batch,
                max_length=max_length
            )

            self.dl_sampler = SameGroupSampler
            self.criterion = TripletLoss(self._model_device)

            # Set model to float() for CLIP
            if hasattr(self.model, "pre_finetuning"):
                self.model.pre_finetuning()

            super().finetune(validation_split=validation_split, epochs=epochs,
                    batch_size=batch_size, optimal_batch_size=optimal_batch_size,
                    early_stopping_epochs=early_stopping_epochs,
                    train_dataloader=self.train_dataloader_triplet,
                    val_dataloader=self.val_dataloader_triplet,
                    dataset_train=dataset_train, dataset_valid=dataset_valid,
                    step=step, configure_optimizers=configure_optimizers)
        
        elif variant == "cosine_similarity":
            texts1 = params["texts1"]
            texts2 = params["texts2"]
            similarity_scores = params["similarity_scores"]

            assert len(texts1) == len(texts2) == len(similarity_scores), "The input lists must match"

            step = step or self.step_cosine

            dataset = TextPairDataset(texts1, texts2, similarity_scores,
                    process_batch=process_batch, max_length=max_length)

            # Set model to float() for CLIP
            if hasattr(self.model, "pre_finetuning"):
                self.model.pre_finetuning()

            super().finetune(dataset=dataset, validation_split=validation_split, epochs=epochs,
                    batch_size=batch_size, optimal_batch_size=optimal_batch_size,
                    early_stopping_epochs=early_stopping_epochs,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    step=step, configure_optimizers=configure_optimizers)