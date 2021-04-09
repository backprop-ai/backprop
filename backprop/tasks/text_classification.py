from typing import List, Tuple, Union, Optional
from backprop.models import BaseModel, AutoModel
from .base import Task
import torch
from transformers.optimization import AdamW
from backprop.utils.datasets import SingleLabelTextClassificationDataset

import requests

DEFAULT_LOCAL_MODEL = "bart-large-mnli"

class TextClassification(Task):
    """
    Task for classification.

    Attributes:
        model:
            1. Model name
            2. Model name on Backprop's text-classification endpoint
            3. Model object that implements the text-classification task
        local (optional): Run locally. Defaults to False
        api_key (optional): Backprop API key for non-local inference
        device (optional): Device to run inference on. Defaults to "cuda" if available.
    """
    def __init__(self, model: Union[str, BaseModel] = None,
                local: bool = False, api_key: str = None, device: str = None):
        task = "text-classification"
        models = AutoModel.list_models(task=task)

        super().__init__(model, local=local, api_key=api_key, device=device,
                        models=models, task=task,
                        default_local_model=DEFAULT_LOCAL_MODEL)

    @staticmethod
    def list_models(return_dict=False, display=False, limit=None):
        return AutoModel.list_models(task="text-classification", return_dict=return_dict, display=display, limit=limit)
    
    def __call__(self, text: Union[str, List[str]], labels: Optional[Union[List[str], List[List[str]]]] = None):
        """Classify input text based on previous training (user-tuned models) or according to given list of labels (zero-shot)

        Args:
            text: string or list of strings to be classified
            labels: list of labels for zero-shot classification (on our out-of-the-box models).
                    If using a user-trained model (e.g. XLNet), this is not used.

        Returns:
            dict where each key is a label and value is probability between 0 and 1, or list of dicts.
        """
        if self.local:
            task = "text-classification"
            
            task_input = {
                "text": text,
                "labels": labels
            }
            if labels:
                return self.model(task_input, task=task)
            else:
                outputs, labels = self.model(task_input, task=task)
                return self.get_label_probabilities(outputs, labels)

        else:
            body = {
                "text": text,
                "labels": labels
            }

            if self.model:
                body["model"] = self.model

            res = requests.post("https://api.backprop.co/text-classification", json=body,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            return res["probabilities"]
    
    def get_label_probabilities(self, outputs, labels):
        is_list = type(outputs) == list

        outputs = outputs if is_list else [outputs]

        probabilities = []
        for o in outputs:
            logits = o[0]
            predictions = torch.softmax(logits, dim=1).detach().squeeze(0).tolist()
            probs = {}
            for idx, pred in enumerate(predictions):
                label = labels[idx]
                probs[label] = pred

            probabilities.append(probs)
        
        probabilities = probabilities if is_list else probabilities[0]

        return probabilities

    def step(self, batch, batch_idx):
        return self.model.training_step(batch, train=True)
    
    def configure_optimizers(self):
        return AdamW(params=self.model.parameters(), lr=2e-5)

    def finetune(self, params, validation_split: Union[float, Tuple[List[int], List[int]]]=0.15,
                 max_input_length: int=128,
                 epochs: int=20, batch_size: int=None, optimal_batch_size: int=None,
                 early_stopping_epochs: int=1, train_dataloader=None, val_dataloader=None, 
                 step=None, configure_optimizers=None):
        """
        I'll do this later.
        """
        inputs = params["texts"]
        outputs = params["labels"]

        assert len(inputs) == len(outputs)

        step = step or self.step
        configure_optimizers = configure_optimizers or self.configure_optimizers

        labels = set(outputs)
        class_to_idx = {v: k for k, v in enumerate(labels)}
        labels = {k: v for k,v in enumerate(labels)}

        output_classes = [class_to_idx[i] for i in outputs]

        if hasattr(self.model, "pre_finetuning"):
            self.model.pre_finetuning(labels)

        print("Processing data...")

        dataset_params = {
            "inputs": inputs,
            "labels": outputs,
            "class_to_idx": class_to_idx,
            "max_input_length": max_input_length
        }

        dataset = SingleLabelTextClassificationDataset(dataset_params, process_batch=self.model.process_batch, length=len(inputs))

        super().finetune(dataset=dataset, validation_split=validation_split, 
                        epochs=epochs, batch_size=batch_size, optimal_batch_size=optimal_batch_size, 
                        early_stopping_epochs=early_stopping_epochs,
                        train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                        step=step, configure_optimizers=configure_optimizers)

