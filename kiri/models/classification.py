from typing import List, Tuple

import requests

DEFAULT_MODEL = "facebook/bart-large-mnli"
DEFAULT_TOKENIZER = "facebook/bart-large-mnli"
model = None
tokenizer = None


def calculate_probability(input_text, label, device):
    hypothesis = f"This example is {label}."
    features = tokenizer.encode(input_text, hypothesis, return_tensors="pt",
                                truncation=True).to(device)
    logits = model(features)[0]
    entail_contradiction_logits = logits[:, [0, 2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    prob_label_is_true = probs[:, 1]
    return prob_label_is_true.item()


def zero_shot(input_text, labels: List[str], model_name: str = None,
              tokenizer_name: str = None, local: bool = False,
              api_key: str = None, device: str = "cpu"):
    # Refer to global variables
    global model
    global tokenizer
    # Setup
    if local:
        # Initialise model
        if model == None:
            from transformers import AutoModelForSequenceClassification
            # Use the default model
            if model_name == None:
                model = AutoModelForSequenceClassification.from_pretrained(
                    DEFAULT_MODEL).to(device)
            # Use the user defined model
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name).to(device)

        # Initialise tokenizer
        if tokenizer == None:
            from transformers import AutoTokenizer
            # Use the default tokenizer
            if tokenizer_name == None:
                tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
            # Use the user defined tokenizer
            else:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if isinstance(input_text, list):
            # Must have a consistent amount of examples
            assert(len(input_text) == len(labels))
            # TODO: implement proper batching
            results_list = []
            for text, labels in zip(input_text, labels):
                results = {}
                for label in labels:
                    results[label] = calculate_probability(text, label, device)

                results_list.append(results)

            return results_list
        else:
            results = {}
            for label in labels:
                results[label] = calculate_probability(
                    input_text, label, device)

            return results

    else:
        if api_key is None:
            raise ValueError(
                "Please provide your api_key (https://kiri.ai) with api_key=... or set local=True")

        body = {
            "text": input_text,
            "labels": labels
        }

        res = requests.post("https://api.kiri.ai/classification", json=body,
                            headers={"x-api-key": api_key}).json()
        return res["probabilities"]
