from typing import List, Tuple
from .generation import generate

import requests


def process_item(question, context, prev_qa):
    input_text = [f"q: {qa[0]} a: {qa[1]}" for qa in prev_qa]
    input_text.append(f"q: {question}")
    input_text.append(f"c: {context}")
    input_text = " ".join(input_text)

    return input_text


def qa(question, context, prev_qa: List[Tuple[str, str]] = [],
       model_name: str = None, tokenizer_name: str = None,
       local: bool = False, api_key: str = None, device: str = "cpu"):
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
        if api_key is None:
            raise ValueError(
                "Please provide your api_key (https://kiri.ai) with api_key=... or set local=True")

        # List of two tuples
        prev_qa = [[q for q, a in prev_qa], [a for q, a in prev_qa]]

        body = {
            "question": question,
            "context": context,
            "prev_q": prev_qa[0],
            "prev_a": prev_qa[1]
        }

        res = requests.post("https://api.kiri.ai/qa", json=body,
                            headers={"x-api-key": api_key}).json()

        return res["answer"]
