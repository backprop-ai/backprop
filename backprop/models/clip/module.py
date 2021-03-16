import torch
from PIL import Image
from typing import Union, List
from . import clip, simple_tokenizer
from kiri.models import PathModel

from io import BytesIO
import base64

class CLIP(PathModel):
    def __init__(self, model_path="ViT-B/32", init_model=clip.load,
                init_tokenizer=simple_tokenizer.SimpleTokenizer, device=None, init=True):
        self.initialised = False
        self.init_model = init_model
        self.init_tokenizer = init_tokenizer
        self.model_path = model_path
        self._device = device

        self.name = "clip"
        self.description = "OpenAI's recently released CLIP model — when supplied with a list of labels and an image, CLIP can accurately predict which labels best fit the provided image."
        self.tasks = ["image-classification"]

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialise
        if init:
            self.model, self.transform = self.init_model(model_path, device=self._device)
            self.tokenizer = self.init_tokenizer()
            
            self.initialised = True

    def __call__(self, task_input, task="image-classification"):
        if task == "image-classification":
            image_base64 = task_input.get("image")
            labels = task_input.get("labels")

            return self.image_classification(image_base64=image_base64, labels=labels)        

    @torch.no_grad()
    def image_classification(self, image_base64: Union[str, List[str]], labels: Union[List[str], List[List[str]]]):
        # TODO: Proper batching
        self.check_init()
        is_list = False

        if type(image_base64) == list:
            is_list = True

        if not is_list:
            image_base64 = [image_base64]
            labels = [labels]

        assert len(image_base64) == len(labels), "images and labels lists must be the same size"
        
        inputs = zip(image_base64, labels)
        probabilities = []

        for image_base64, labels in inputs:

            # Not bytes
            if type(image_base64) == str:
                image_base64 = image_base64.split(",")[-1]

            image = BytesIO(base64.b64decode(image_base64))

            image = self.transform(Image.open(image)).unsqueeze(0).to(self._device)
            text = clip.tokenize(self.tokenizer, labels).to(self._device)

            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            
            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy().tolist()[0]

            label_probs = zip(labels, probs)
            probabilities.append({lp[0]: lp[1] for lp in label_probs})

        if is_list == False:
            probabilities = probabilities[0]

        return probabilities