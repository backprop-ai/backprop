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

DEFAULT_LOCAL_MODEL = "clip"

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
        task = "image-classification"
        models = AutoModel.list_models(task=task)

        super().__init__(model, local=local, api_key=api_key, device=device,
                        models=models, task=task,
                        default_local_model=DEFAULT_LOCAL_MODEL)

    
    def __call__(self, image_path: Union[str, List[str]], labels: Union[List[str], List[List[str]]] = None):
        """Classify image according to given labels.

        Args:
            image_path: path to image or list of paths to image
            labels: list of strings or list of labels (for zero shot classification)

        Returns:
            dict where each key is a label and value is probability between 0 and 1 or list of dicts
        """

        is_list = False

        if type(image_path) == list:
            is_list = True

        if not is_list:
            image_path = [image_path]

        image = []
        for img in image_path:
            if not isinstance(img, Image.Image):
                with open(img, "rb") as image_file:
                    img = base64.b64encode(image_file.read())
            else:
                buffered = BytesIO()
                img.save(buffered, format=img.format)
                img = base64.b64encode(buffered.getvalue())
            
            image.append(img)

        if not is_list:
            image = image[0]
            
        if self.local:
            task_input = {
                "image": image,
                "labels": labels
            }
            return self.model(task_input, task="image-classification")
        else:
            body = {
                "image": image,
                "labels": labels,
                "model": self.model,
            }

            res = requests.post("https://api.backprop.co/image-classification", json=body,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            return res["probabilities"]

    
    def step_single_label(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images, task="image-classification", train=True)

        loss = self.criterion(outputs, targets)
        return loss

    def step_multi_label(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images, task="image-classification", train=True)

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