from typing import List, Tuple

DEFAULT_MODEL = "msmarco-distilroberta-base-v2"
model = None


def vectorise(input_text, model_name: str = None,
              local: bool = True):
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
                model = SentenceTransformer(model_name)

        return model.encode(input_text)

    else:
        raise ValueError("Non local inference is not implemented!")
