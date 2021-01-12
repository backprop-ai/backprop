from typing import List, Tuple
from .generation import generate

import requests


def process_item(item):
    return f"emotion: {item}"


def emotion(input_text,
            model_name: str = None, tokenizer_name: str = None,
            local: bool = False, api_key: str = None, device: str = "cpu"):
    if local:
        if isinstance(input_text, list):
            # Process according to the model used
            input_text = [process_item(item) for item in input_text]
        else:
            input_text = process_item(input_text)

        return generate(input_text, model_name=model_name,
                        tokenizer_name=tokenizer_name, local=local)
    else:
        if api_key is None:
            raise ValueError(
                "Please provide your api_key (https://kiri.ai) with api_key=... or set local=True")

        body = {
            "text": input_text
        }

        res = requests.post("https://api.kiri.ai/emotion", json=body,
                            headers={"x-api-key": api_key}).json()

        return res["emotion"]
