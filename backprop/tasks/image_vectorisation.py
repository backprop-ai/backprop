from typing import List, Tuple, Union, Dict
from .base import Task
from backprop.models import BaseModel, AutoModel
import base64
from PIL import Image
from io import BytesIO

import requests
from backprop.utils import img_to_base64, path_to_img
from backprop.utils.samplers import SameGroupSampler
from backprop.utils.losses import TripletLoss
import torch
import torch.nn.functional as F
import random
from backprop.utils.datasets import ImageGroupDataset, ImagePairDataset
from torch.utils.data.dataloader import DataLoader
import os

TASK = "image-vectorisation"

DEFAULT_LOCAL_MODEL = "clip"

LOCAL_ALIASES = {
    "clip": "clip-vit-b32"
}

class ImageVectorisation(Task):
    """
    Task for image vectorisation.

    Attributes:
        model:
            1. Model name
            2. Model name on Backprop's image-vectorisation endpoint
            3. Model object that implements the image-vectorisation task
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
    
    def __call__(self, image: Union[Union[str, Image.Image], Union[List[str], List[Image.Image]]],
                return_tensor=False):
        """Vectorise input image.

        Args:
            image: image or list of images to vectorise. Can be both PIL Image objects or paths to images.

        Returns:
            Vector or list of vectors
        """
        is_list = False

        vector = None
        image = path_to_img(image)

        task_input = {
            "image": image
        }

        if self.local:
            vector = self.model(task_input, task=TASK,
                                return_tensor=return_tensor)
        else:
            task_input["image"] = img_to_base64(task_input["image"])
            task_input["model"] = self.model

            res = requests.post("https://api.backprop.co/image-vectorisation", json=task_input,
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
        image, group = batch

        img_vecs_norm = self.model({"image": image}, task=TASK,
                    return_tensor=True, preprocess=False, train=True)

        loss = self.criterion(img_vecs_norm, group)

        return loss

    def step_cosine(self, batch, batch_idx):
        imgs1, imgs2, similarity_scores = batch

        img_vecs1_norm = self.model({"image": imgs1}, task=TASK,
                    return_tensor=True, preprocess=False, train=True)
        img_vecs2_norm = self.model({"image": imgs2}, task=TASK,
                    return_tensor=True, preprocess=False, train=True)

        loss = torch.cosine_similarity(img_vecs1_norm, img_vecs2_norm)
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
        optimal_batch_size = getattr(self.model, "optimal_batch_size", 128)

        configure_optimizers = configure_optimizers or self.configure_optimizers

        if variant == "triplet":
            images = params["images"]
            groups = params["groups"]
            assert len(images) == len(groups), "The input lists must match"

            step = step or self.step_triplet

            if isinstance(validation_split, tuple):
                train_idx, val_idx = validation_split
            else:
                all_idx = list(range(len(images)))
                val_len = int(len(all_idx) * validation_split)
                val_idx = random.sample(all_idx, val_len)
                train_idx = list(set(all_idx) - set(val_idx))
            
            dataset_train = ImageGroupDataset(
                [images[i] for i in train_idx],
                [groups[i] for i in train_idx],
                self.model.process_image,
            )

            dataset_valid = ImageGroupDataset(
                [images[i] for i in val_idx],
                [groups[i] for i in val_idx],
                self.model.process_image,
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
            imgs1 = params["imgs1"]
            imgs2 = params["imgs2"]
            similarity_scores = params["similarity_scores"]

            assert len(imgs1) == len(imgs2) == len(similarity_scores), "The input lists must match"

            step = step or self.step_cosine

            dataset = ImagePairDataset(imgs1, imgs2, similarity_scores,
                    self.model.process_image)

            # Set model to float() for CLIP
            if hasattr(self.model, "pre_finetuning"):
                self.model.pre_finetuning()

            super().finetune(dataset=dataset, validation_split=validation_split, epochs=epochs,
                    batch_size=batch_size, optimal_batch_size=optimal_batch_size,
                    early_stopping_epochs=early_stopping_epochs,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    step=step, configure_optimizers=configure_optimizers)