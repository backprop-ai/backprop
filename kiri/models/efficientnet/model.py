import torch
from PIL import Image
from typing import Union, List
from efficientnet_pytorch import EfficientNet as EfficientNet_pt
from torchvision import transforms
from kiri.models import PathModel
import json
import os

from io import BytesIO
import base64

class EfficientNet(PathModel):
    def __init__(self, model_path="efficientnet-b4", init_model=EfficientNet_pt.from_pretrained,
                init_tokenizer=None, device=None, init=True):
        name = "efficientnet-b4"
        description = "EfficientNet is an image classification model that achieves state-of-the-art accuracy while being an order-of-magnitude smaller and faster than previous models. Trained on ImageNet's 1000 categories."
        tasks = ["image-classification"]
        self.image_size = EfficientNet_pt.get_image_size(model_path)
        
        with open(os.path.join(os.path.dirname(__file__), "imagenet_labels.txt"), "r") as f:
            self.labels = json.load(f)

        PathModel.__init__(self, model_path, init_model, name=name, description=description,
            tasks=tasks)

    def __call__(self, task_input, task="image-classification"):
        if task == "image-classification":
            image_base64 = task_input.get("image")

            return self.image_classification(image_base64=image_base64)        

    @torch.no_grad()
    def image_classification(self, image_base64: Union[str, List[str]], top_k=10):
        # TODO: Proper batching
        self.check_init()
        is_list = False

        if type(image_base64) == list:
            is_list = True

        if not is_list:
            image_base64 = [image_base64]
        
        probabilities = []

        tfms = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=Image.BICUBIC),
                transforms.CenterCrop(self.image_size),
                lambda image: image.convert("RGB"),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        for image_base64 in image_base64:

            # Not bytes
            if type(image_base64) == str:
                image_base64 = image_base64.split(",")[-1]

            image = BytesIO(base64.b64decode(image_base64))
            image = Image.open(image)

            image = tfms(image).unsqueeze(0).to(self._device)

            logits = self.model(image)
            preds = torch.topk(logits, k=top_k).indices.squeeze(0).tolist()

            probs = {}
            for idx in preds:
                label = self.labels[str(idx)]
                prob = torch.softmax(logits, dim=1)[0, idx].item()

                probs[label] = prob

            probabilities.append(probs)                

        if is_list == False:
            probabilities = probabilities[0]

        return probabilities