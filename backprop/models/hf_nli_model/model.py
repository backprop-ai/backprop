from typing import List, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from backprop.models import HFModel

class HFNLIModel(HFModel):
    def __init__(self, model_path=None, tokenizer_path=None, name: str = None,
                description: str = None, tasks: List[str] = None, details: Dict = None,
                model_class=AutoModelForSequenceClassification,
                tokenizer_class=AutoTokenizer, device=None):
        tasks = tasks or ["text-classification"]
        
        HFModel.__init__(self, model_path, name=name, description=description,
                    tasks=tasks, details=details, tokenizer_path=tokenizer_path,
                    model_class=model_class, tokenizer_class=tokenizer_class,
                    device=device)

    def __call__(self, task_input, task="text-classification"):
        if task == "text-classification":
            text = task_input.get("text")
            labels = task_input.get("labels")

            if labels == None:
                raise ValueError("labels must be provided")

            return self.classify(text, labels)
        else:
            raise ValueError(f"Unsupported task: {task}")

    @staticmethod
    def list_models():
        from .models_list import models

        return models

    def calculate_probability(self, text, label, device):
        hypothesis = f"This example is {label}."
        features = self.tokenizer.encode(text, hypothesis, return_tensors="pt",
                                    truncation=True).to(self._model_device)
        logits = self.model(features)[0]
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:, 1]
        return prob_label_is_true.item()

    def classify(self, text, labels):
        """
        Classifies text, given a set of labels.
        """
        if isinstance(text, list):
            # Must have a consistent amount of examples
            assert(len(text) == len(labels))
            # TODO: implement proper batching
            results_list = []
            for text, labels in zip(text, labels):
                results = {}
                for label in labels:
                    results[label] = self.calculate_probability(text, label, self._model_device)

                results_list.append(results)

            return results_list
        else:
            results = {}
            for label in labels:
                results[label] = self.calculate_probability(
                    text, label, self._model_device)

            return results