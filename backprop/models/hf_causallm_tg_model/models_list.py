gpt2_large = {
    "description": "A large (774M parameter) version of OpenAI's GPT-2 model. This is a general-use model, and has not been further finetuned on any specific languages or tasks.",
    "tasks": ["text-generation"],
    "init_kwargs": {
        "model_path": "gpt2-large"
    },
    "details": {
        "num_parameters": 407344131,
    }
}


models = {
    "gpt2-large": gpt2_large,
}