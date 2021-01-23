from transformers import AutoModelForPreTraining, AutoTokenizer
import torch

import requests

DEFAULT_MODEL = "gpt2"
DEFAULT_TOKENIZER = "gpt2"

MODELS = {
    "gpt2": DEFAULT_MODEL,
    "t5-base-qa-summary-emotion": "kiri-ai/t5-base-qa-summary-emotion"
}

TOKENIZERS = {
    "gpt2": DEFAULT_TOKENIZER,
    "t5-base-qa-summary-emotion": "kiri-ai/t5-base-qa-summary-emotion"
}


def generate(model, text, local: bool = True, 
            api_key: str = None, **kwargs):
    if local:
        # Name according to huggingface
        kwargs["num_return_sequences"] = kwargs.pop("num_generations", 1)

        return model_generate(model, tokenizer, device, input_text, **kwargs)
    else:
        if api_key is None:
            raise ValueError(
                "Please provide your api_key (https://kiri.ai) with api_key=... or set local=True")

        if model_name == None:
            model_name = "gpt2-large"

        body = {
            "text": input_text,
            "model": model_name
        }

        body = {**body, **kwargs}
        
        res = requests.post("https://api.kiri.ai/generation", json=body,
                            headers={"x-api-key": api_key}).json()
        return res["output"]