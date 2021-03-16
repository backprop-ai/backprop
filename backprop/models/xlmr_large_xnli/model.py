from transformers import AutoModelForSequenceClassification, AutoTokenizer
from kiri.models import HuggingModel

class XLMRLargeXNLI(HuggingModel):
    def __init__(self, model_path="joeddav/xlm-roberta-large-xnli", tokenizer_path=None,
                model_class=AutoModelForSequenceClassification,
                tokenizer_class=AutoTokenizer, device=None, init=True):
        HuggingModel.__init__(self, model_path, tokenizer_path=tokenizer_path,
                    model_class=model_class, tokenizer_class=tokenizer_class,
                    device=device, init=init)
        self.name = "xlmr-large-xnli"
        self.description = "XLM-RoBERTa is a multilingual variant of Facebook's RoBERTa model. This has been finetuned on the XNLI dataset, resulting in classification system that is effective on 100 different languages."
        self.tasks = ["text-classification"]

    def __call__(self, task_input, task="text-classification"):
        if task in ["text-classification", "classification"]:
            text = task_input.get("text")
            labels = task_input.get("labels")
            return self.classify(text, labels)
        else:
            raise ValueError(f"Unsupported task: {task}")


    def calculate_probability(self, text, label, device):
        hypothesis = f"This example is {label}."
        features = self.tokenizer.encode(text, hypothesis, return_tensors="pt",
                                    truncation=True).to(self._device)
        logits = self.model(features)[0]
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:, 1]
        return prob_label_is_true.item()

    def classify(self, text, labels):
        """
        Classifies text, given a set of labels.
        """
        self.check_init()
        if isinstance(text, list):
            # Must have a consistent amount of examples
            assert(len(text) == len(labels))
            # TODO: implement proper batching
            results_list = []
            for text, labels in zip(text, labels):
                results = {}
                for label in labels:
                    results[label] = self.calculate_probability(text, label, self._device)

                results_list.append(results)

            return results_list
        else:
            results = {}
            for label in labels:
                results[label] = self.calculate_probability(
                    text, label, self._device)

            return results