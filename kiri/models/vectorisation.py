from typing import List, Tuple

import requests

DEFAULT_MODEL = "msmarco-distilroberta-base-v2"

MODELS = {
    "english": DEFAULT_MODEL,
    "multilingual": "distiluse-base-multilingual-cased-v2"
}

model = None


def vectorise(input_text, model_name: str = None,
              local: bool = False, api_key: str = None, device: str = "cpu"):
    # Refer to global variables
    global model
    # Setup
    if local:
        # Initialise model
        if model == None:
            from sentence_transformers import SentenceTransformer
            # Use the default model
            if model_name == None:
                model = SentenceTransformer(
                    DEFAULT_MODEL)
            # Use the user defined model
            else:
                # Get from predefined list or try to find remotely
                model_name = MODELS.get(model_name) or model_name
                model = SentenceTransformer(model_name)

        return model.encode(input_text)

    else:
        if api_key is None:
            raise ValueError(
                "Please provide your api_key (https://kiri.ai) with api_key=... or set local=True")

        if model_name == None:
            model_name = "english"

        body = {
            "text": input_text,
            "model": model_name
        }

        res = requests.post("https://api.kiri.ai/vectorisation", json=body,
                            headers={"x-api-key": api_key}).json()

        return res["vector"]
