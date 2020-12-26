from typing import List, Tuple

DEFAULT_MODEL = "facebook/bart-large-mnli"
DEFAULT_TOKENIZER = "facebook/bart-large-mnli"
model = None
tokenizer = None


def zero_shot(input_text, labels: List[str], model_name: str = None,
              tokenizer_name: str = None, local: bool = True):
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
                    DEFAULT_MODEL)
            # Use the user defined model
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name)

        # Initialise tokenizer
        if tokenizer == None:
            from transformers import AutoTokenizer
            # Use the default tokenizer
            if tokenizer_name == None:
                tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
            # Use the user defined tokenizer
            else:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        results = {}
        for label in labels:
            hypothesis = f"This example is {label}."
            features = tokenizer.encode(input_text, hypothesis, return_tensors="pt",
                                        truncation_strategy="only_first")
            logits = model(features)[0]
            entail_contradiction_logits = logits[:, [0, 2]]
            probs = entail_contradiction_logits.softmax(dim=1)
            prob_label_is_true = probs[:, 1]
            results[label] = prob_label_is_true.item()

        return results

    else:
        raise ValueError("Non local inference is not implemented!")
