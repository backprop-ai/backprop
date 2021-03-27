import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from PIL import Image
from typing import Union, List
from functools import partial
from . import clip, simple_tokenizer
from backprop.models import PathModel, Finetunable
from backprop.utils import ImageTextPairDataset, base64_to_img

from io import BytesIO
import base64

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
        self.tasks = ["image-classification", "image-vectorisation", "text-vectorisation"]

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

        img_text_vecs = torch.cat([text_vecs, image_vecs], 1)
        img_text_vecs /= img_text_vecs.norm(dim=-1, keepdim=True)

        return img_text_vecs

    def configure_optimizers(self):
        return torch.optim.AdamW(params=self.model.parameters(), lr=1e-5)

    def common_step(self, batch, batch_idx):
        texts1, imgs1, texts2, imgs2, similarity_scores = batch

        text_vecs1 = self.model.encode_text(texts1)
        text_vecs2 = self.model.encode_text(texts2)
        img_vecs1 = self.model.encode_image(imgs1)
        img_vecs2 = self.model.encode_image(imgs2)

        # Combine vecs
        img_text_vecs1 = torch.cat([text_vecs1, img_vecs1], 1)
        img_text_vecs2 = torch.cat([text_vecs2, img_vecs2], 1)

        # Normalize
        img_text_vecs1_norm = img_text_vecs1 / img_text_vecs1.norm(dim=-1, keepdim=True)
        img_text_vecs2_norm = img_text_vecs2 / img_text_vecs2.norm(dim=-1, keepdim=True)

        loss = torch.cosine_similarity(img_text_vecs1_norm, img_text_vecs2_norm)
        loss = F.mse_loss(loss, similarity_scores.view(-1))

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def finetune(self, params,
                 validation_split: float=0.15, epochs: int=20,
                 batch_size: int=None, early_stopping: bool = True,
                 trainer: pl.Trainer = None, task: str = "image-text-vectorisation"):
        if task == "image-text-vectorisation":
            img_text_pairs1 = params["img_text_pairs1"]
            img_text_pairs2 = params["img_text_pairs2"]
            similarity_scores = params["similarity_scores"]
            assert len(img_text_pairs1) == len(img_text_pairs2) == len(similarity_scores), "The input lists must match"
            
            dataset = ImageTextPairDataset(img_text_pairs1, img_text_pairs2, similarity_scores,
                    self.transform, self.tokenizer)
        else:
            raise ValueError(f"Unsupported task: {task}")


        OPTIMAL_BATCH_SIZE = 128

        self.model.float()

        Finetunable.finetune(self, dataset, validation_split=validation_split,
            epochs=epochs, batch_size=batch_size, optimal_batch_size=OPTIMAL_BATCH_SIZE,
            early_stopping=early_stopping, trainer=trainer)