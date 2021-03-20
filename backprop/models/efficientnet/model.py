import torch
from PIL import Image
from typing import Union, List
from efficientnet_pytorch import EfficientNet as EfficientNet_pt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from random import shuffle
from backprop.models import PathModel, Finetunable
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

class EfficientNet(PathModel, Finetunable):
    """
    EfficientNet is a very efficient image-classification model. Trained on ImageNet.

    Attributes:
        model_path: Any efficientnet model (smaller to bigger) from efficientnet-b0 to efficientnet-b7
        init_model: Callable that initialises the model from the model_path
        kwargs: kwrags passed to :class:`backprop.models.generic_models.PathModel`
    """
    def __init__(self, model_path: str = "efficientnet-b0", init_model = None, **kwargs):
        Finetunable.__init__(self)
        
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

        PathModel.__init__(self, model_path, init_model, **kwargs)
        
        self.name = model_path
        self.description = "EfficientNet is an image classification model that achieves state-of-the-art accuracy while being an order-of-magnitude smaller and faster than previous models. Trained on ImageNet's 1000 categories."
        self.tasks = ["image-classification"]

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

    @torch.no_grad()
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

    def configure_optimizers(self):
        return torch.optim.SGD(params=self.model.parameters(), lr=1e-1, weight_decay=1e-4)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, targets)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def finetune(self, image_dir: str, validation_split: float = 0.15, epochs: int = 20,
                batch_size: int = None, early_stopping: bool = True,
                trainer: pl.Trainer = None):
        """
        Finetunes EfficientNet for the image-classification task.
        
        Note:
            ``image_dir`` has a strict structure that must be followed:

            .. code-block:: text
            
                Images
                ├── Cool_Dog
                │   ├── dog1.jpg
                │   └── dog2.jpg
                └── Amazing_Dog
                    ├── dog1.jpg
                    └── dog2.jpg
        
            In the example above, our ``image_dir`` is called Images. It contains two classes, each of which have 2 training examples.
            Every class must have its own folder, every folder must have some images as examples.

        Args:
            image_dir: Path to your training data
            validation_split: Float between 0 and 1 that determines what percentage of the data to use for validation
            epochs: Integer that specifies how many iterations of training to do
            batch_size: Leave as None to determine the batch size automatically
            early_stopping: Boolean that determines whether to automatically stop when validation loss stops improving
            trainer: Your custom pytorch_lightning trainer

        Examples::

            import backprop
            
            # Initialise model
            model = backprop.models.EfficientNet()

            # Finetune with path to your images
            model.finetune("my_image_dir")
        """
        OPTIMAL_BATCH_SIZE = 128
        
        dataset = ImageFolder(image_dir, transform=self.tfms)
        self.labels = {str(v): k for k, v in dataset.class_to_idx.items()}
        num_classes = len(dataset.classes)

        if self.num_classes != num_classes:
            self.num_classes = num_classes
            self.model = EfficientNet_pt.from_pretrained(self.model_path, num_classes=num_classes)

        Finetunable.finetune(self, dataset, validation_split=validation_split,
            epochs=epochs, optimal_batch_size=OPTIMAL_BATCH_SIZE, batch_size=batch_size,
            early_stopping=early_stopping, trainer=trainer)
