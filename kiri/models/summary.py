from typing import List, Tuple
from .generation import generate


def summarise(input_text,
              model_name: str = None, tokenizer_name: str = None,
              local: bool = True):
    if local:
        # Process according to the model used
        input_text = f"summarise: {input_text}"

        return generate(input_text, model_name=model_name,
                        tokenizer_name=tokenizer_name, local=local)
    else:
        raise ValueError("Non local inference is not implemented!")
