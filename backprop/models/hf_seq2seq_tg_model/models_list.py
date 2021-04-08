t5_small = {
    "description": "This is the T5 model by Google.",
    "tasks": ["text-generation"],
    "init_kwargs": {
        "model_path": "t5-small"
    },
    "details": {
        "num_parameters": 407344131,
    }
}


models = {
    "t5-small": t5_small,
}