from typing import List, Tuple

DEFAULT_MODEL = "kiri-ai/t5-base-qa-summary-emotion"
DEFAULT_TOKENIZER = "t5-base"
model = None
tokenizer = None


def generate(input_text, model_name: str = None, tokenizer_name: str = None,
             local: bool = True, device: str = "cpu"):
    # Refer to global variables
    global model
    global tokenizer
    # Setup
    if local:
        # Initialise model
        if model == None:
            from transformers import T5ForConditionalGeneration
            # Use the default model
            if model_name == None:
                model = T5ForConditionalGeneration.from_pretrained(
                    DEFAULT_MODEL)
            # Use the user defined model
            else:
                model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Initialise tokenizer
        if tokenizer == None:
            from transformers import T5Tokenizer
            # Use the default tokenizer
            if tokenizer_name == None:
                tokenizer = T5Tokenizer.from_pretrained(DEFAULT_TOKENIZER)
            # Use the user defined tokenizer
            else:
                tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)

        is_list = False
        if isinstance(input_text, list):
            is_list = True

        features = tokenizer(input_text, padding=True, return_tensors='pt')
        tokens = model.generate(input_ids=features['input_ids'],
                                attention_mask=features['attention_mask'], max_length=128)
        if is_list:
            return [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in tokens]
        else:
            return tokenizer.decode(tokens[0], skip_special_tokens=True)

    else:
        raise ValueError("Non local inference is not implemented!")
