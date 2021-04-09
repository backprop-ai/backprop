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

DEFAULT_LOCAL_MODEL = "msmarco-distilroberta-base-v2"

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

        task = "text-vectorisation"
        models = AutoModel.list_models(task=task)

        super().__init__(model, local=local, api_key=api_key, device=device,
                        models=models, task=task,
                        default_local_model=DEFAULT_LOCAL_MODEL)

    @staticmethod
    def list_models(return_dict=False, display=False, limit=None):
        return AutoModel.list_models(task="image-vectorisation", return_dict=return_dict, display=display, limit=limit)
    
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
            vector = self.model(task_input, task="text-vectorisation")
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
        return torch.optim.AdamW(params=self.model.parameters(), lr=1e-5)

    def step_triplet(self, batch, batch_idx):
        text, group = batch

        text_vecs_norm = self.model({"text": text}, task="text-vectorisation",
                    return_tensor=True, preprocess=False, train=True)

        loss = self.criterion(text_vecs_norm, group)

        return loss

    def step_cosine(self, batch, batch_idx):
        texts1, texts2, similarity_scores = batch

        text_vecs1_norm = self.model({"text": texts1}, task="text-vectorisation",
                    return_tensor=True, preprocess=False, train=True)
        text_vecs2_norm = self.model({"text": texts2}, task="text-vectorisation",
                    return_tensor=True, preprocess=False, train=True)

        loss = torch.cosine_similarity(text_vecs1_norm, text_vecs2_norm)
        loss = F.mse_loss(loss, similarity_scores.view(-1))

        return loss

    def train_dataloader_triplet(self):
        return DataLoader(self.dataset_train,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() or 0,
            sampler=self.dl_sampler(self.dataset_train))

    def val_dataloader_triplet(self):
        return DataLoader(self.dataset_valid,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() or 0,
            sampler=self.dl_sampler(self.dataset_valid))

    def finetune(self, params, validation_split: Union[float, Tuple[List[int], List[int]]] = 0.15,
                max_length: int = None, variant: str = "triplet", epochs: int = 20, batch_size: int = None,
                optimal_batch_size: int = None, early_stopping_epochs: int = 1,
                train_dataloader = None, val_dataloader = None, step = None, configure_optimizers = None):
        optimal_batch_size = getattr(self.model, "optimal_batch_size", 128)

        configure_optimizers = configure_optimizers or self.configure_optimizers

        process_text = partial(self.model.process_text, max_length=max_length)

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
                process_text,
            )

            dataset_valid = TextGroupDataset(
                [texts[i] for i in val_idx],
                [groups[i] for i in val_idx],
                process_text,
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
                    process_text)

            # Set model to float() for CLIP
            if hasattr(self.model, "pre_finetuning"):
                self.model.pre_finetuning()

            super().finetune(dataset=dataset, validation_split=validation_split, epochs=epochs,
                    batch_size=batch_size, optimal_batch_size=optimal_batch_size,
                    early_stopping_epochs=early_stopping_epochs,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    step=step, configure_optimizers=configure_optimizers)