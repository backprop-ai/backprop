from typing import List, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from backprop.models import HFModel
import torch

class HFNLIModel(HFModel):
    """
    Class for Hugging Face sequence classification models trained on a NLI dataset

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

    @torch.no_grad()
    def __call__(self, task_input, task="text-classification"):
        """
        Uses the model for the text-classification task

        Args:
            task_input: input dictionary according to the ``text-classification`` task specification.
                Needs labels (for zero-shot).
            task: text-classification
        """
        if task == "text-classification":
            is_list = False

            text = task_input.get("text")
            labels = task_input.get("labels")

            if labels == None:
                raise ValueError("labels must be provided")

            if isinstance(text, list):
                is_list = True
            else:
                text = [text]
                labels = [labels]

            # Must have a consistent amount of examples
            assert(len(text) == len(labels))

            probs = self.classify(text, labels)

            if not is_list:
                probs = probs[0]

            return probs
        else:
            raise ValueError(f"Unsupported task: {task}")

    @staticmethod
    def list_models():
        from .models_list import models

        return models

    def calculate_probability(self, text, labels):
        batch_features = []
        
        hypothesis = [f"This example is {l}." for l in labels]
        features = self.tokenizer([text]*len(hypothesis), hypothesis, return_tensors="pt",
                                    truncation=True, padding=True).to(self._model_device)

        logits = self.model(features["input_ids"], features["attention_mask"])[0]
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:, 1]
        return prob_label_is_true.tolist()

    def classify(self, text, labels):
        """
        Classifies text, given a set of labels.
        """
        results_list = []
        for text, labels in zip(text, labels):
            results = {}
            probs = self.calculate_probability(text, labels)

            for prob, label in zip(probs, labels):
                results[label] = prob

            results_list.append(results)

        return results_list