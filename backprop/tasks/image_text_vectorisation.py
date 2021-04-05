from typing import List, Tuple, Union
from .base import Task
from backprop.models import CLIP, BaseModel
from backprop.utils import path_to_img, img_to_base64
from backprop.utils import ImageTextGroupDataset, base64_to_img
from backprop.utils.losses import TripletLoss
import base64
from PIL import Image
from io import BytesIO
import torch
import random

import requests
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler
import os
import numpy as np

DEFAULT_LOCAL_MODEL = CLIP

LOCAL_MODELS = {
    "clip": DEFAULT_LOCAL_MODEL,
}

DEFAULT_API_MODEL = "clip"

API_MODELS = ["clip"]

FINETUNABLE_MODELS = []

class ImageTextVectorisation(Task):
    """
    Task for combined imag-text vectorisation.

    Attributes:
        model:
            1. Name of the model on Backprop's vectorisation endpoint (clip or your own uploaded model)
            2. Officially supported local models (clip).
            3. Model class of instance Backprop's BaseModel (that supports the task)
            4. Path/name of saved Backprop model
        local (optional): Run locally. Defaults to False
        api_key (optional): Backprop API key for non-local inference
        device (optional): Device to run inference on. Defaults to "cuda" if available.
    """
    def __init__(self, model: Union[str, BaseModel] = None,
                local: bool = False, api_key: str = None, device: str = None):

        super().__init__(model, local=local, api_key=api_key, device=device,
                        local_models=LOCAL_MODELS, api_models=API_MODELS,
                        default_local_model=DEFAULT_LOCAL_MODEL,
                        default_api_model=DEFAULT_API_MODEL)
    
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
            vector = self.model(task_input, task="image-text-vectorisation",
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

        img_text_vecs_norm = self.model({"image": image, "text": text}, task="image-text-vectorisation",
                    return_tensor=True, preprocess=False, train=True)

        loss = self.criterion(img_text_vecs_norm, group)

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

    def finetune(self, params, validation_split: Union[float, Tuple[List[int], List[int]]],
                variant: str = "triplet", epochs: int = 20, batch_size: int = None,
                optimal_batch_size: int = None, early_stopping_epochs: int = 1,
                train_dataloader = None, val_dataloader = None, step = None):
        if variant == "triplet":
            images = params["images"]
            texts = params["texts"]
            groups = params["groups"]
            assert len(images) == len(texts) == len(groups), "The input lists must match"

            if isinstance(validation_split, tuple):
                train_idx, val_idx = validation_split
            else:
                all_idx = list(range(len(images)))
                val_len = int(len(all_idx) * validation_split)
                val_idx = random.sample(val_len, all_idx)
                train_idx = list(set(all_idx) - set(val_idx))
            
            dataset_train = ImageTextGroupDataset(
                [images[i] for i in train_idx],
                [texts[i] for i in train_idx],
                [groups[i] for i in train_idx],
                self.model.process_image,
                self.model.process_text
            )

            dataset_valid = ImageTextGroupDataset(
                [images[i] for i in val_idx],
                [texts[i] for i in val_idx],
                [groups[i] for i in val_idx],
                self.model.process_image,
                self.model.process_text
            )

            self.dl_sampler = SameGroupSampler
            self.criterion = TripletLoss(self._model_device)

            # Make this a part of CLIP somehow
            self.model.model.float()

            super().finetune(validation_split=validation_split, epochs=epochs,
                    batch_size=batch_size, optimal_batch_size=self.model.optimal_batch_size or 128,
                    early_stopping_epochs=early_stopping_epochs,
                    train_dataloader=self.train_dataloader_triplet,
                    val_dataloader=self.val_dataloader_triplet,
                    dataset_train=dataset_train, dataset_valid=dataset_valid,
                    step=self.step_triplet)



class SameGroupSampler(Sampler):
    def __init__(self, dataset):
        super().__init__(dataset)

        groups = dataset.groups

        items = zip(list(range(len(groups))), groups)

        item_to_group = {}
        group_to_items = {}

        for idx, group in items:
            item_to_group[idx] = group

            if group not in group_to_items:
                group_to_items[group] = [idx]
            else:
                group_to_items[group].append(idx)

        self.groups = set(groups)
        self.item_to_group = item_to_group
        self.group_to_items = group_to_items
        
    def __len__(self):
        return len(self.groups)
        
    def __iter__(self):
        for _ in range(len(self)):
            # Sample one group
            group_sample = random.sample(self.groups, 1)[0]
            
            items = self.group_to_items[group_sample]
            replace = False
            if len(items) < 2:
                replace = True

            # Sample two ids
            sample1, sample2 = np.random.choice(items, 2, replace=replace)
            
            yield sample1
            yield sample2