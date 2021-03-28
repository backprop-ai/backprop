import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
import pytorch_lightning as pl
import numpy as np

from PIL import Image
from typing import Union, List
from functools import partial
from . import clip, simple_tokenizer
from backprop.models import PathModel, Finetunable
from backprop.utils import ImageTextGroupDataset, base64_to_img
from backprop.utils.losses import TripletLoss

from io import BytesIO
import base64
import random
from torch.utils.data.dataloader import DataLoader
import os

class CLIP(PathModel, Finetunable):
    def __init__(self, model_path="ViT-B/32", init_model=clip.load,
                init_tokenizer=simple_tokenizer.SimpleTokenizer, device=None):
        Finetunable.__init__(self)
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
            
    def __call__(self, task_input, task="image-classification", return_tensor=False):
        output = None
        is_list = False
        
        if task == "image-classification":
            image = task_input.get("image")
            labels = task_input.get("labels")

            image = base64_to_img(image)

            if type(image) == list:
                is_list = True
            else:
                image = [image]
                labels = [labels]
            
            assert len(image) == len(labels), "images and labels lists must be the same size"

            output = self.image_classification(image=image, labels=labels)

        elif task == "image-vectorisation":
            image = task_input.get("image")
            image = base64_to_img(image)

            if type(image) == list:
                is_list = True
            else:
                image = [image]

            img_vecs = self.image_vectorisation(image=image) 

            if not return_tensor:
                img_vecs = img_vecs.tolist()

            output = img_vecs

        elif task == "text-vectorisation":
            text = task_input.get("text")

            if type(text) == list:
                is_list = True
            else:
                text = [text]

            text_vecs = self.text_vectorisation(text=text) 

            if not return_tensor:
                text_vecs = text_vecs.tolist()

            output = text_vecs
        
        elif task == "image-text-vectorisation":
            image = task_input.get("image")
            text = task_input.get("text")

            if type(image) == list:
                is_list = True
            else:
                image = [image]
                text = [text]

            assert len(image) == len(text), "image and text lists must be the same size"

            img_text_vecs = self.image_text_vectorisation(image, text)

            if not return_tensor:
                img_text_vecs = img_text_vecs.tolist()
            
            output = img_text_vecs

        if not is_list:
            output = output[0]

        return output

    @torch.no_grad()
    def image_classification(self, image: List[Image.Image], labels: List[List[str]]):
        # TODO: Proper batching
        
        inputs = zip(image, labels)

        probabilities = []
        for image, labels in inputs:

            image = self.transform(image.unsqueeze(0)).to(self._model_device)

            text = self.tokenizer(labels).to(self._model_device)
            
            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy().tolist()[0]

            label_probs = zip(labels, probs)
            probabilities.append({lp[0]: lp[1] for lp in label_probs})

        return probabilities

    @torch.no_grad()
    def image_vectorisation(self, image: List[Image.Image]):
        image = [self.transform(img) for img in image]
        image = torch.stack(image).to(self._model_device)

        image_features = self.model.encode_image(image)

        return image_features

    @torch.no_grad()
    def text_vectorisation(self, text: List[str]):
        text = self.tokenizer(text).to(self._model_device)

        text = self.model.encode_text(text)

        return text

    @torch.no_grad()
    def image_text_vectorisation(self, image: List[Image.Image], text: List[str]):
        image_vecs = self.image_vectorisation(image)
        text_vecs = self.text_vectorisation(text)

        img_text_vecs = torch.cat([image_vecs, text_vecs], 1)
        img_text_vecs /= img_text_vecs.norm(dim=-1, keepdim=True)

        return img_text_vecs

    def configure_optimizers(self):
        return torch.optim.AdamW(params=self.model.parameters(), lr=1e-5)

    def common_step(self, batch, batch_idx):
        image, text, group = batch

        img_vecs = self.model.encode_image(image)
        text_vecs = self.model.encode_text(text)

        # Combine vecs
        img_text_vecs = torch.cat([img_vecs, text_vecs], 1)

        # Normalize
        img_text_vecs_norm = img_text_vecs / img_text_vecs.norm(dim=-1, keepdim=True)

        loss = self.criterion(img_text_vecs_norm, group)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() or 0,
            sampler=self.dl_sampler(self.dataset_train) or None)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() or 0,
            sampler=self.dl_sampler(self.dataset_valid) or None)

    def finetune(self, params,
                 validation_split: float=0.15, train_idx: List[int] = None,
                 val_idx: List[int] = None, epochs: int=20,
                 batch_size: int=None, early_stopping: bool = True,
                 trainer: pl.Trainer = None, task: str = "image-text-vectorisation"):
        if task == "image-text-vectorisation":
            images = params["images"]
            texts = params["texts"]
            groups = params["groups"]
            assert len(images) == len(texts) == len(groups), "The input lists must match"
            
            if train_idx and val_idx:
                dataset_train = ImageTextGroupDataset(
                    [images[i] for i in train_idx],
                    [texts[i] for i in train_idx],
                    [groups[i] for i in train_idx],
                    self.transform,
                    self.tokenizer
                )

                dataset_valid = ImageTextGroupDataset(
                    [images[i] for i in val_idx],
                    [texts[i] for i in val_idx],
                    [groups[i] for i in val_idx],
                    self.transform,
                    self.tokenizer
                )

                self.dataset_train = dataset_train
                self.dataset_valid = dataset_valid

            dataset = ImageTextGroupDataset(images, texts, groups,
                    self.transform, self.tokenizer)
            self.dl_sampler = SameGroupSampler
            self.criterion = TripletLoss(self._model_device)
        else:
            raise ValueError(f"Unsupported task: {task}")

        OPTIMAL_BATCH_SIZE = 128

        self.model.float()

        Finetunable.finetune(self, dataset, validation_split=validation_split,
            epochs=epochs, batch_size=batch_size, optimal_batch_size=OPTIMAL_BATCH_SIZE,
            early_stopping=early_stopping, trainer=trainer)

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