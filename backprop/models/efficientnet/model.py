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

IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/backprop-ai/backprop/main/backprop/models/efficientnet/imagenet_labels.txt"

class EfficientNet(PathModel):
    """
    EfficientNet is a very efficient image-classification model. Trained on ImageNet.

    Attributes:
        model_path: Any efficientnet model (smaller to bigger) from efficientnet-b0 to efficientnet-b7
        init_model: Callable that initialises the model from the model_path
        kwargs: kwrags passed to :class:`backprop.models.generic_models.PathModel`
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
    
    @torch.no_grad
    def __call__(self, task_input, task="image-classification"):
        """
        Uses the model for the image-classification task

        Args:
            task_input: input dictionary according to the ``image-classification`` task specification
            task: image-classification
        """
        if task == "image-classification":
            image_base64 = task_input.get("image")

            return self.image_classification(image_base64=image_base64)

    def pre_finetuning(self, labels=None, num_classes=None):
        self.labels = labels
        
        if self.num_classes != num_classes:
            self.num_classes = num_classes
            self.model = EfficientNet_pt.from_pretrained(self.model_path, num_classes=num_classes)

    def training_step(self, batch, task="image-classification"):
        return self.model(batch)

    def image_classification(self, image_base64: Union[str, List[str]], top_k=10):
        # TODO: Proper batching
        is_list = False

        if type(image_base64) == list:
            is_list = True

        if not is_list:
            image_base64 = [image_base64]
        
        probabilities = []

        for image_base64 in image_base64:

            # Not bytes
            if type(image_base64) == str:
                image_base64 = image_base64.split(",")[-1]

            image = BytesIO(base64.b64decode(image_base64))
            image = Image.open(image)

            image = self.tfms(image).unsqueeze(0).to(self._model_device)

            logits = self.model(image)
            preds = torch.topk(logits, k=top_k).indices.squeeze(0).tolist()
            dist = torch.softmax(logits, dim=1)
            probs = {}
            for idx in preds:
                label = self.labels[str(idx)]
                prob = dist[0, idx].item()

                probs[label] = prob

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

