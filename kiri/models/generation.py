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

model = None
tokenizer = None
m_checkpoint = None
t_checkpoint = None

def model_generate(model, tokenizer, device, input_text, **kwargs):
    # Get and remove do_sample or set to False
    do_sample = kwargs.pop("do_sample", None) or False
    params = ["temperature", "top_k", "top_p", "repetition_penalty",
                "length_penalty", "num_beams", "num_return_sequences"]

    # If params are changed, we want to sample
    for param in params:
        if param in kwargs.keys():
            do_sample = True
            break

    is_list = False
    if isinstance(input_text, list):
        is_list = True

    if not is_list:
        input_text = [input_text]


    all_tokens = []
    for text in input_text:
        features = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            tokens = model.generate(
                input_ids=features["input_ids"].to(device),
                attention_mask=features["attention_mask"].to(device),
                do_sample=do_sample,
                **kwargs)

            all_tokens.append(tokens)

    value = []
    for tokens in all_tokens:
        value.append([tokenizer.decode(tokens, skip_special_tokens=True)
                 for tokens in tokens])
    
    output = value

    # Unwrap generation list
    if kwargs.get("num_return_sequences", 1) == 1:
        output_unwrapped = []
        for value in output:
            output_unwrapped.append(value[0])

        output = output_unwrapped
    
    # Return single item
    if not is_list:
        output = output[0]

    return output

def generate(input_text, model_name: str = None, 
               tokenizer_name: str = None, local: bool = True, 
               api_key: str = None, device: str = "cpu", do_sample=False, **kwargs):

    global model
    global tokenizer
    global m_checkpoint
    global t_checkpoint
    
    if local:
        # Initialise model, allow switching if new one is provided.
        if not model or (model_name and model_name != m_checkpoint):
            if not model_name:
                model = AutoModelForPreTraining.from_pretrained(DEFAULT_MODEL)
                model_name = DEFAULT_MODEL
            else:
                # Get from predefined list or try to find remotely
                model_name = MODELS.get(model_name) or model_name
                model = AutoModelForPreTraining.from_pretrained(model_name)
        
        model.to(device)
        
        # Initialise tokenizer, allow switching if new one is provided.
        if not tokenizer or (tokenizer_name and tokenizer_name != t_checkpoint):
            if not tokenizer_name:
                tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER, use_fast=False)
            else:
                tokenizer_name = TOKENIZERS.get(tokenizer_name) or tokenizer_name
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        
        # Set the checkpoints used to load the model/tokenizer
        m_checkpoint = model_name
        t_checkpoint = tokenizer_name

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






