gpt2_large = {
    "description": "A large (774M parameter) version of OpenAI's GPT-2 model. This is a general-use model, and has not been further finetuned on any specific languages or tasks.",
    "tasks": ["text-generation"],
    "init_kwargs": {
        "model_path": "gpt2-large"
    },
    "details": {
        "num_parameters": 774030080,
    }
}

gpt2_medium = {
    "description": "A medium version of OpenAI's GPT-2 model. This is a general-use model, and has not been further finetuned on any specific languages or tasks.",
    "tasks": ["text-generation"],
    "init_kwargs": {
        "model_path": "gpt2-medium"
    },
    "details": {
        "num_parameters": 354823168,
    }
}

distilgpt2 = {
    "description": "A distilled (82M parameter) version of OpenAI's GPT-2 model.",
    "tasks": ["text-generation"],
    "init_kwargs": {
        "model_path": "distilgpt2"
    },
    "details": {
        "num_parameters": 81912576,
    }
}


models = {
    "gpt2-large": gpt2_large,
    "gpt2-medium": gpt2_medium,
    "distilgpt2": distilgpt2
}