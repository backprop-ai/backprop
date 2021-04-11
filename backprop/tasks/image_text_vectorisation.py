from typing import List, Tuple, Union, Dict
from .base import Task
from backprop.models import AutoModel, BaseModel
from backprop.utils import path_to_img, img_to_base64
from backprop.utils import ImageTextGroupDataset, base64_to_img, ImageTextPairDataset
from backprop.utils.losses import TripletLoss
from backprop.utils.samplers import SameGroupSampler
import base64
from PIL import Image
from io import BytesIO
import torch
import random

import requests
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import os
import numpy as np

TASK = "image-text-vectorisation"

DEFAULT_LOCAL_MODEL = "clip"

LOCAL_ALIASES = {
    "clip": "clip-vit-b32"
}

class ImageTextVectorisation(Task):
    """
    Task for combined image-text vectorisation.

    Attributes:
        model:
            1. Model name
            2. Model name on Backprop's image-text-vectorisation endpoint
            3. Model object that implements the image-text-vectorisation task
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
        return AutoModel.list_models(task=TASK, return_dict=return_dict, display=display, limit=limit, aliases=LOCAL_ALIASES)

    def __call__(self, image: Union[str, List[str]], text: Union[str, List[str]], return_tensor=False):
        """Vectorise input image and text pairs.

        Args:
            image: image or list of images to vectorise. Can be both PIL Image objects or paths to images.
            text: text or list of text to vectorise. Must match image ordering.

        Returns:
            Vector or list of vectors
        """
        vector = None
        image = path_to_img(image)

        if self.local:

            task_input = {
                "image": image,
                "text": text
            }
            vector = self.model(task_input, task=TASK,
                                return_tensor=return_tensor)
        else:
            raise NotImplementedError("This task is not yet implemented in the API")

            # image = img_to_base64(image)

            # body = {
            #     "image": image,
            #     "model": self.model
            # }

            # res = requests.post("https://api.backprop.co/image-vectorisation", json=body,
            #                     headers={"x-api-key": self.api_key}).json()

            # if res.get("message"):
            #     raise Exception(f"Failed to make API request: {res['message']}")

            # vector = res["vector"]

        if return_tensor and not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector)

        return vector

    def configure_optimizers(self):
        return torch.optim.AdamW(params=self.model.parameters(), lr=1e-5)

    def step_triplet(self, batch, batch_idx):
        image, text, group = batch

        img_text_vecs_norm = self.model.training_step({"image": image, "text": text}, task=TASK)

        loss = self.criterion(img_text_vecs_norm, group)

        return loss

    def step_cosine(self, batch, batch_idx):
        texts1, imgs1, texts2, imgs2, similarity_scores = batch

        img_text_vecs1_norm = self.model.training_step({"image": imgs1, "text": texts1}, task=TASK)
        img_text_vecs2_norm = self.model.training_step({"image": imgs2, "text": texts2}, task=TASK)

        loss = torch.cosine_similarity(img_text_vecs1_norm, img_text_vecs2_norm)
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
                variant: str = "triplet", epochs: int = 20, batch_size: int = None,
                optimal_batch_size: int = None, early_stopping_epochs: int = 1,
                train_dataloader = None, val_dataloader = None, step = None, configure_optimizers = None):
        """
        Finetunes a model for combined image & text vectorisation. Includes different variants for calculating loss.
        
        Args:
            dataset: Torch dataset on which training will occur.
            validation_split: Float between 0 and 1 that determines percentage of data to use for validation.
            variant: How loss will be calculated: "triplet" (default) or "cosine_similarity".
            epochs: Integer specifying how many training iterations to run.
            batch_size: Batch size when training. Leave as None to automatically determine batch size.
            optimal_batch_size: 
            early_stopping_epochs: Integer determining how many epochs will run before stopping without an improvement in validation loss.
            train_dataloader: Dataloader for providing training data when finetuning. Defaults to inbuilt dataloder.
            val_dataloader: Dataloader for providing validation data when finetuning. Defaults to inbuilt dataloader.
            step: Function determining how to call model for a training step. Defaults to step defined in this task class.
            configure_optimizers: Function that sets up the optimizer for training. Defaults to optimizer defined in this task class.
        """

        optimal_batch_size = getattr(self.model, "optimal_batch_size", 128)

        configure_optimizers = configure_optimizers or self.configure_optimizers

        if variant == "triplet":
            images = params["images"]
            texts = params["texts"]
            groups = params["groups"]
            assert len(images) == len(texts) == len(groups), "The input lists must match"

            step = step or self.step_triplet

            if isinstance(validation_split, tuple):
                train_idx, val_idx = validation_split
            else:
                all_idx = list(range(len(images)))
                val_len = int(len(all_idx) * validation_split)
                val_idx = random.sample(all_idx, val_len)
                train_idx = list(set(all_idx) - set(val_idx))
            
            dataset_train = ImageTextGroupDataset(
                [images[i] for i in train_idx],
                [texts[i] for i in train_idx],
                [groups[i] for i in train_idx],
                self.model.process_batch
            )

            dataset_valid = ImageTextGroupDataset(
                [images[i] for i in val_idx],
                [texts[i] for i in val_idx],
                [groups[i] for i in val_idx],
                self.model.process_batch
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
            img_text_pairs1 = params["img_text_pairs1"]
            img_text_pairs2 = params["img_text_pairs2"]
            similarity_scores = params["similarity_scores"]

            assert len(img_text_pairs1) == len(img_text_pairs2) == len(similarity_scores), "The input lists must match"

            step = step or self.step_cosine

            dataset = ImageTextPairDataset(img_text_pairs1, img_text_pairs2, similarity_scores, self.model.process_batch)

            # Set model to float() for CLIP
            if hasattr(self.model, "pre_finetuning"):
                self.model.pre_finetuning()

            super().finetune(dataset=dataset, validation_split=validation_split, epochs=epochs,
                    batch_size=batch_size, optimal_batch_size=optimal_batch_size,
                    early_stopping_epochs=early_stopping_epochs,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    step=step, configure_optimizers=configure_optimizers)