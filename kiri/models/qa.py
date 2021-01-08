from typing import List, Tuple
from .generation import generate


def process_item(question, context, prev_qa):
    input_text = [f"q: {qa[0]} a: {qa[1]}" for qa in prev_qa]
    input_text.append(f"q: {question}")
    input_text.append(f"c: {context}")
    input_text = " ".join(input_text)

    return input_text


def qa(question, context, prev_qa: List[Tuple[str, str]] = [],
       model_name: str = None, tokenizer_name: str = None,
       local: bool = True):
    if local:
        if isinstance(question, list):
            # Must have a consistent amount of examples
            assert(len(question) == len(context))
            if len(prev_qa) != 0:
                assert(len(question) == len(prev_qa))
            else:
                prev_qa = [prev_qa] * len(question)

            # Process according to the model used
            input_text = [process_item(q, c, p)
                          for q, c, p in zip(question, context, prev_qa)]
        else:
            input_text = process_item(question, context, prev_qa)

        return generate(input_text, model_name=model_name,
                        tokenizer_name=tokenizer_name, local=local)
    else:
        raise ValueError("Non local inference is not implemented!")
