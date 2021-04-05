import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
import pytorch_lightning as pl
import numpy as np

from PIL import Image
from typing import Union, List
from functools import partial
from . import clip, simple_tokenizer
from backprop.models import BaseModel, Finetunable
from backprop.utils import ImageTextGroupDataset, base64_to_img
from backprop.utils.losses import TripletLoss

from io import BytesIO
import base64
import random
from torch.utils.data.dataloader import DataLoader
import os

class CLIP(BaseModel):
    def __init__(self, model_path="ViT-B/32", init_model=clip.load,
                init_tokenizer=simple_tokenizer.SimpleTokenizer, device=None):
        BaseModel.__init__(self, None)
        self.init_model = init_model
        self.init_tokenizer = init_tokenizer
        self.model_path = model_path
        self._model_device = device

        self.name = "clip"
        self.description = "OpenAI's recently released CLIP model — when supplied with a list of labels and an image, CLIP can accurately predict which labels best fit the provided image."
        self.tasks = ["image-classification", "image-vectorisation", "text-vectorisation", "image-text-vectorisation"]

        if self._model_device is None:
            self._model_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialise
        self.model, self.transform = self.init_model(model_path, device=self._model_device)
        tokenizer = self.init_tokenizer()
        self.tokenizer = partial(clip.tokenize, tokenizer)
        self.process_text = self.tokenizer
        self.process_image = self.transform
        self.optimal_batch_size = 128
            
    def __call__(self, task_input, task="image-classification", return_tensor=False, preprocess=True, train=False):
        output = None
        is_list = False
        
        if task == "image-classification":
            image = task_input.get("image")
            labels = task_input.get("labels")

            if preprocess:
                image = base64_to_img(image)

                if type(image) == list:
                    is_list = True
                else:
                    image = [image]
                    labels = [labels]
            
                assert len(image) == len(labels), "images and labels lists must be the same size"

                image = [self.process_image(img).unsqueeze(0).to(self._model_device) for img in image]

                text = [self.process_text(l).to(self._model_device) for l in labels]
            else:
                is_list = True
            
            with torch.set_grad_enabled(train):
                output = self.image_classification(image=image, text=text, labels=labels)

        elif task == "image-vectorisation":
            image = task_input.get("image")

            if preprocess:
                image = base64_to_img(image)
                
                if type(image) == list:
                    is_list = True
                else:
                    image = [image]

                image = [self.process_image(img) for img in image]
                image = torch.stack(image).to(self._model_device)
            else:
                is_list = True

            with torch.set_grad_enabled(train):
                img_vecs = self.image_vectorisation(image=image) 

            if not return_tensor:
                img_vecs = img_vecs.tolist()

            output = img_vecs

        elif task == "text-vectorisation":
            text = task_input.get("text")

            if preprocess:
                if type(text) == list:
                    is_list = True
                else:
                    text = [text]

                text = self.tokenizer(text).to(self._model_device)
            else:
                is_list = True

            with torch.set_grad_enabled(train):
                text_vecs = self.text_vectorisation(text=text)

            if not return_tensor:
                text_vecs = text_vecs.tolist()

            output = text_vecs
        
        elif task == "image-text-vectorisation":
            image = task_input.get("image")
            text = task_input.get("text")

            if preprocess:
                image = base64_to_img(image)

                if type(image) == list:
                    is_list = True
                else:
                    image = [image]
                    text = [text]

                assert len(image) == len(text), "image and text lists must be the same size"

                text = self.tokenizer(text).to(self._model_device)
                image = [self.process_image(img) for img in image]
                image = torch.stack(image).to(self._model_device)
            else:
                is_list = True

            # with torch.set_grad_enabled(train):
            img_text_vecs = self.image_text_vectorisation(image, text)

            if not return_tensor:
                img_text_vecs = img_text_vecs.tolist()
            
            output = img_text_vecs

        if not is_list:
            output = output[0]

        return output


    def image_classification(self, image: torch.TensorType, text: torch.TensorType, labels):
        probabilities = []
        inputs = zip(image, text, labels)
        for image, text, labels in inputs:
            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy().tolist()[0]

            label_probs = zip(labels, probs)
            probabilities.append({lp[0]: lp[1] for lp in label_probs})

        return probabilities

    def image_vectorisation(self, image: torch.TensorType):
        image_features = self.model.encode_image(image)

        return image_features

    def text_vectorisation(self, text: torch.TensorType):
        text = self.model.encode_text(text)

        return text

    def image_text_vectorisation(self, image: List[Image.Image], text: List[str]):
        image_vecs = self.model.encode_image(image)
        text_vecs = self.model.encode_text(text)

        img_text_vecs = torch.cat([image_vecs, text_vecs], 1)
        img_text_vecs_norm = img_text_vecs / img_text_vecs.norm(dim=-1, keepdim=True)

        return img_text_vecs_norm

