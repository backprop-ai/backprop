from typing import List, Tuple
from .generation import generate


def qa(question, context, prev_qa: List[Tuple[str, str]] = [],
       model_name: str = None, tokenizer_name: str = None,
       local: bool = True):
    if local:
        # Process according to the model used
        input_text = [f"q: {qa[0]} a: {qa[1]}" for qa in prev_qa]
        input_text.append(f"q: {question}")
        input_text.append(f"c: {context}")
        input_text = " ".join(input_text)
        return generate(input_text, model_name=model_name,
                        tokenizer_name=tokenizer_name, local=local)
    else:
        raise ValueError("Non local inference is not implemented!")
