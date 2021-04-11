import torch
from PIL import Image
from typing import Union, List, Dict
from efficientnet_pytorch import EfficientNet as EfficientNet_pt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from random import shuffle
from backprop.models import PathModel
from backprop.utils.download import download
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from functools import partial
import torch.nn.functional as F
import pytorch_lightning as pl
import json
import os

from io import BytesIO
import base64
from backprop.utils.helpers import base64_to_img

IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/backprop-ai/backprop/main/backprop/models/efficientnet/imagenet_labels.txt"

class EfficientNet(PathModel):
    """
    EfficientNet is a very efficient image-classification model. Trained on ImageNet.

    Attributes:
        model_path: Any efficientnet model (smaller to bigger) from efficientnet-b0 to efficientnet-b7
        init_model: Callable that initialises the model from the model_path
        name: string identifier for the model. Lowercase letters and numbers.
            No spaces/special characters except dashes.
        description: String description of the model.
        tasks: List of supported task strings
        details: Dictionary of additional details about the model
        device: Device for model. Defaults to "cuda" if available.
    """
    def __init__(self, model_path: str = "efficientnet-b0", init_model = None, name: str = None,
                description: str = None, tasks: List[str] = None, details: Dict = None,
                device=None):
        self.image_size = EfficientNet_pt.get_image_size(model_path)
        self.num_classes = 1000

        if init_model is None:
            init_model = partial(EfficientNet_pt.from_pretrained, num_classes=self.num_classes)
        
        with open(download(IMAGENET_LABELS_URL, "efficientnet"), "r") as f:
            self.labels = json.load(f)
            self.labels = {int(k): v for k, v in self.labels.items()}

        self.tfms = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=Image.BICUBIC),
                transforms.CenterCrop(self.image_size),
                lambda image: image.convert("RGB"),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        tasks = tasks or ["image-classification"]

        self.optimal_batch_size = 128
        self.process_image = self.tfms
        
        PathModel.__init__(self, model_path, name=name, description=description,
                            details=details, tasks=tasks, init_model=init_model,
                            device=device)

    @staticmethod
    def list_models():
        from .models_list import models

        return models
    
    @torch.no_grad()
    def __call__(self, task_input, task="image-classification"):
        """
        Uses the model for the image-classification task

        Args:
            task_input: input dictionary according to the ``image-classification`` task specification
            task: image-classification
        """

        if task == "image-classification":
            image = task_input.get("image")
            top_k = task_input.get("top_k", 10000)

            image = base64_to_img(image)

            return self.image_classification(image=image, top_k=top_k)

    def pre_finetuning(self, labels=None, num_classes=None):
        self.labels = labels
        
        if self.num_classes != num_classes:
            self.num_classes = num_classes
            self.model = EfficientNet_pt.from_pretrained(self.model_path, num_classes=num_classes)

    def training_step(self, batch, task="image-classification"):
        return self.model(batch)

    def image_classification(self, image, top_k=10000):
        # TODO: Proper batching
        is_list = False

        if type(image) == list:
            is_list = True

        if not is_list:
            image = [image]
        
        probabilities = []

        for image in image:
            image = self.tfms(image).unsqueeze(0).to(self._model_device)

            logits = self.model(image)

            # TODO: Handle multi label
            dist = torch.softmax(logits, dim=1).squeeze(0).tolist()
            
            probs = {}
            for idx, prob in enumerate(dist):
                label = self.labels[idx]

                probs[label] = prob

            probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            probs = {k: v for k, v in probs[:top_k]}

            probabilities.append(probs)                

        if is_list == False:
            probabilities = probabilities[0]

        return probabilities

    def process_batch(self, params, task="image-classification"):
        image = params["image"]
        image = Image.open(image)
        image = self.process_image(image).squeeze(0)
        return image

    def configure_optimizers(self):
        return torch.optim.SGD(params=self.model.parameters(), lr=1e-1, weight_decay=1e-4)

