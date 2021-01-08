from typing import List, Tuple
from .generation import generate


def process_item(item):
    return f"summarise: {item}"


def summarise(input_text,
              model_name: str = None, tokenizer_name: str = None,
              local: bool = True):
    if local:
        if isinstance(input_text, list):
            # Process according to the model used
            input_text = [process_item(item) for item in input_text]
        else:
            input_text = process_item(input_text)

        return generate(input_text, model_name=model_name,
                        tokenizer_name=tokenizer_name, local=local)
    else:
        raise ValueError("Non local inference is not implemented!")
