from typing import List, Tuple, Union
from ..models import BaseModel
from backprop.models import AutoModel
from backprop.utils.datasets import SingleLabelImageClassificationDataset, MultiLabelImageClassificationDataset
from .base import Task
from PIL import Image
from torch import nn
import torch
import numpy as np
import base64

import requests
from io import BytesIO
from backprop.utils.helpers import img_to_base64, path_to_img

TASK = "image-classification"

DEFAULT_LOCAL_MODEL = "clip"

LOCAL_ALIASES = {}

class ImageClassification(Task):
    """
    Task for image classification.

    Attributes:
        model:
            1. Model name
            2. Model name on Backprop's image-classification endpoint
            3. Model object that implements the image-classification task
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
        return AutoModel.list_models(task=TASK, return_dict=return_dict, display=display, limit=limit)

    
    def __call__(self, image: Union[str, List[str]], labels: Union[List[str], List[List[str]]] = None,
                top_k: int = 0, top_p: int = 1.0, multi_label: bool = False):
        """Classify image according to given labels.

        Args:
            image: image or list of images to vectorise. Can be both PIL Image objects or paths to images.
            labels: list of strings or list of labels (for zero shot classification)
            top_k: return probabilities only for top_k predictions. Use 0 to get all.
            top_p: return probabilities that sum to top_p (between 0 and 1). Use 1.0 to get all.
            multi_label: allow multiple true labels. Not all models support this.

        Returns:
            dict where each key is a label and value is probability between 0 and 1 or list of dicts
        """

        image = path_to_img(image)
            
        task_input = {
            "image": image,
            "labels": labels,
            "top_k": top_k,
            "top_p": top_p,
            "multi_label": multi_label
        }

        if self.local:
            return self.model(task_input, task=TASK)
        else:
            task_input["image"] = img_to_base64(task_input["image"])
            task_input["model"] = self.model

            res = requests.post("https://api.backprop.co/image-classification", json=task_input,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            return res["probabilities"]

    
    def step_single_label(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images, task=TASK, train=True)

        loss = self.criterion(outputs, targets)
        return loss

    def step_multi_label(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images, task=TASK, train=True)

        loss = self.criterion(outputs, targets)
        return loss
    

    def finetune(self, params, validation_split: Union[float, Tuple[List[int], List[int]]]=0.15,
                 variant: str = "single_label",
                 epochs: int=20, batch_size: int=None, optimal_batch_size: int=None,
                 early_stopping_epochs: int=1, train_dataloader=None, val_dataloader=None, 
                 step=None, configure_optimizers=None):
        """
        TODO
        """
        optimal_batch_size = optimal_batch_size or getattr(self.model, "optimal_batch_size", 128)

        configure_optimizers = configure_optimizers or self.model.configure_optimizers or self.configure_optimizers

        images = params["images"]
        labels = params["labels"]

        assert len(images) == len(labels), "The input lists must match"
        
        if variant == "single_label":

            step = step or self.step_single_label

            labels_set = set(labels)
            labels_dict = {k: v for k, v in enumerate(labels_set)}

            if hasattr(self.model, "pre_finetuning"):
                self.model.pre_finetuning(labels=labels_dict, num_classes=len(labels_set))

            dataset = SingleLabelImageClassificationDataset(images, labels, self.model.process_image)

            self.criterion = nn.CrossEntropyLoss()
        elif variant == "multi_label":
            step = step or self.step_multi_label

            all_labels = set(np.concatenate(labels).flat)

            labels_dict = {i: label for i, label in enumerate(all_labels)}

            if hasattr(self.model, "pre_finetuning"):
                self.model.pre_finetuning(labels=labels_dict, num_classes=len(all_labels))

            dataset = MultiLabelImageClassificationDataset(images, labels, self.model.process_image)

            # Sigmoid and BCE
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported variant '{variant}'")

        super().finetune(dataset=dataset, validation_split=validation_split, 
                        epochs=epochs, batch_size=batch_size, optimal_batch_size=optimal_batch_size, 
                        early_stopping_epochs=early_stopping_epochs,
                        train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                        step=step, configure_optimizers=configure_optimizers)