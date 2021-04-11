import torch
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import AdamW
from backprop.models import HFModel
from typing import List, Union, Dict


class HFSeqTCModel(HFModel):
    """
    Class for Hugging Face sequence classification models

    Attributes:
        model_path: path to HF model
        tokenizer_path: path to HF tokenizer
        name: string identifier for the model. Lowercase letters and numbers.
            No spaces/special characters except dashes.
        description: String description of the model.
        tasks: List of supported task strings
        details: Dictionary of additional details about the model
        model_class: Class used to initialise model
        tokenizer_class: Class used to initialise tokenizer
        device: Device for model. Defaults to "cuda" if available.
    """

    def __init__(self, model_path=None, tokenizer_path=None, name: str = None,
                description: str = None, tasks: List[str] = None, details: Dict = None,
                model_class=AutoModelForSequenceClassification,
                tokenizer_class=AutoTokenizer, device=None):
        tasks = tasks or ["text-classification"]

        HFModel.__init__(self, model_path, name=name, description=description,
                    tasks=tasks, details=details, tokenizer_path=tokenizer_path,
                    model_class=model_class, tokenizer_class=tokenizer_class,
                    device=device)

        self.pre_finetuning = self.init_pre_finetune

    @staticmethod
    def list_models():
        from .models_list import models

        return models

    @torch.no_grad()
    def __call__(self, task_input, task="text-classification", train=False):
        """
        Uses the model for text classification.
        At this point, the model needs to already have been finetuned.
        This is what sets up the final layer for classification.

        Args:
            task_input: input dictionary according to the ``text-classification`` task specification
            task: text-classification
        """
        if task == "text-classification":
            text = task_input.pop("text")
            top_k = task_input.get("top_k", 10000)

            is_list = type(text) == list
            
            text = text if is_list else [text]

            outputs = []
            for t in text:
                tokens = self.tokenizer(t, truncation=True, padding=True, return_tensors="pt")
                input_ids = tokens.input_ids[0].unsqueeze(0).to(self._model_device)
                mask = tokens.attention_mask[0].unsqueeze(0).to(self._model_device)
                
                output = self.model(input_ids=input_ids, attention_mask=mask)
                outputs.append(output)
            
            outputs = outputs if is_list else outputs[0]
            
            return self.get_label_probabilities(outputs, top_k)
        else:
            raise ValueError(f"Unsupported task: {task}")


    def get_label_probabilities(self, outputs, top_k):
        is_list = type(outputs) == list

        outputs = outputs if is_list else [outputs]

        probabilities = []
        for o in outputs:
            logits = o[0]
            predictions = torch.softmax(logits, dim=1).detach().squeeze(0).tolist()
            probs = {}
            for idx, pred in enumerate(predictions):
                label = self.labels[idx]
                probs[label] = pred

            probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            probs = {k: v for k, v in probs[:top_k]}

            probabilities.append(probs)
        
        probabilities = probabilities if is_list else probabilities[0]

        return probabilities
    
    def training_step(self, batch, task="text-classification"):
        return self.model(**batch)[0]

    def process_batch(self, params, task="text-classification"):
        text = params["inputs"]
        class_to_idx = params["class_to_idx"]
        target = class_to_idx[params["labels"]]

        tokens = self.tokenizer(text, truncation=True, max_length=params["max_length"], padding="max_length", return_tensors="pt")
        return {
            "input_ids": tokens.input_ids[0],
            "attention_mask": tokens.attention_mask[0],
            "labels": target
        }

    def encode(self, text, target, max_input_length=128):
        tokens = self.tokenizer(text, truncation=True, max_length=max_input_length, padding="max_length", return_tensors="pt")
        return tokens.input_ids[0], tokens.attention_mask[0], target

    def init_pre_finetune(self, labels):
        if not hasattr(self, "labels") or len(self.labels) != len(labels):
            self.labels = labels
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=len(labels))
