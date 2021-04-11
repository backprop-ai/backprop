import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
import pytorch_lightning as pl
import numpy as np

from PIL import Image
from typing import Union, List, Dict
from functools import partial
from . import clip, simple_tokenizer
from backprop.models import BaseModel
from backprop.utils import ImageTextGroupDataset, base64_to_img
from backprop.utils.losses import TripletLoss

from io import BytesIO
import base64
import random
from torch.utils.data.dataloader import DataLoader
import os

class CLIP(BaseModel):
    def __init__(self, model_path="ViT-B/32", init_model=clip.load,
                init_tokenizer=simple_tokenizer.SimpleTokenizer, name: str = None,
                description: str = None, tasks: List[str] = None, details: Dict = None,
                device=None):
        BaseModel.__init__(self, None, name=name, description=description, tasks=tasks, details=details)
        self.init_model = init_model
        self.init_tokenizer = init_tokenizer
        self.model_path = model_path
        self._model_device = device

        if self._model_device is None:
            self._model_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialise
        self.model, self.transform = self.init_model(model_path, device=self._model_device)
        tokenizer = self.init_tokenizer()
        self.tokenizer = partial(clip.tokenize, tokenizer)
        self.process_image = self.transform
        self.optimal_batch_size = 128
        # Can't specify max_length
        self.max_length = None
        self.pre_finetuning = self.model.float

    @staticmethod
    def list_models():
        from .models_list import models

        return models
            
    @torch.no_grad()
    def __call__(self, task_input, task="image-classification", return_tensor=False):
        output = None
        is_list = False
        
        if task == "image-classification":
            image = task_input.get("image")
            labels = task_input.get("labels")
            top_k = task_input.get("top_k")

            image = base64_to_img(image)

            if labels is None:
                raise ValueError("labels must be provided")

            is_list = type(image) == list
            if not is_list:
                image = [image]
                labels = [labels]

            image = [self.process_image(img).unsqueeze(0).to(self._model_device) for img in image]
            text = [self.tokenizer(l).to(self._model_device) for l in labels]
            
            output = self.image_classification(image=image, text=text, labels=labels, top_k=top_k)

        elif task == "image-vectorisation":
            image = task_input.get("image")

            image = base64_to_img(image)

            is_list = type(image) == list
            if not is_list:
                image = [image]

            image = [self.process_image(img) for img in image]
            image = torch.stack(image).to(self._model_device)

            img_vecs = self.image_vectorisation(image=image) 

            if not return_tensor:
                img_vecs = img_vecs.tolist()

            output = img_vecs

        elif task == "text-vectorisation":
            text = task_input.get("text")

            is_list = type(text) == list
            if not is_list:
                text = [text]

            text = self.tokenizer(text).to(self._model_device)

            text_vecs = self.text_vectorisation(text=text)

            if not return_tensor:
                text_vecs = text_vecs.tolist()

            output = text_vecs
        
        elif task == "image-text-vectorisation":
            image = task_input.get("image")
            text = task_input.get("text")

            image = base64_to_img(image)
            
            is_list = type(image) == list
            if not is_list:
                image = [image]
                text = [text]

            text = self.tokenizer(text).to(self._model_device)
            image = [self.process_image(img) for img in image]
            image = torch.stack(image).to(self._model_device)

            img_text_vecs = self.image_text_vectorisation(image, text)

            if not return_tensor:
                img_text_vecs = img_text_vecs.tolist()
            
            output = img_text_vecs

        if not is_list:
            output = output[0]

        return output

    def training_step(self, params, task):
        if task == "image-vectorisation":
            image = params["image"]
            return self.image_vectorisation(image)
        elif task == "text-vectorisation":
            text = params["text"]
            return self.text_vectorisation(text)
        elif task == "image-text-vectorisation":
            image = params["image"]
            text = params["text"]
            return self.image_text_vectorisation(image, text)

    def process_batch(self, params, task):
        if task == "image-vectorisation":
            image = params["image"]
            return self.process_image(Image.open(image)).squeeze(0)
        elif task == "text-vectorisation":
            text = params["text"]
            return self.tokenizer(text)
    
    def image_classification(self, image: torch.TensorType, text: torch.TensorType, labels, top_k=10000):
        probabilities = []
        inputs = zip(image, text, labels)


        for image, text, labels in inputs:
            # Don't exceed the number of labels
            top_k = min(len(labels), top_k)
            
            logits_per_image, logits_per_text = self.model(image, text)
            
            probs = logits_per_image.softmax(dim=-1)
            idx = torch.topk(probs, k=top_k, sorted=True).indices.squeeze(0).tolist()
            probs = probs.tolist()[0]

            labels = [labels[i] for i in idx]
            probs = [probs[i] for i in idx]

            label_probs = zip(labels, probs)
            probabilities.append({lp[0]: lp[1] for lp in label_probs})

        return probabilities

    def image_vectorisation(self, image: torch.TensorType):
        image_features = self.model.encode_image(image)

        return image_features

    def text_vectorisation(self, text: torch.TensorType):
        text = self.model.encode_text(text)

        return text

    def image_text_vectorisation(self, image: torch.TensorType, text: torch.TensorType):
        image_vecs = self.model.encode_image(image)
        text_vecs = self.model.encode_text(text)

        img_text_vecs = torch.cat([image_vecs, text_vecs], 1)
        img_text_vecs_norm = img_text_vecs / img_text_vecs.norm(dim=-1, keepdim=True)

        return img_text_vecs_norm

