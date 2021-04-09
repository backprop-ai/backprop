t5_small = {
    "description": "This is the small T5 model by Google.",
    "tasks": ["text-generation"],
    "init_kwargs": {
        "model_path": "t5-small"
    },
    "details": {
        "num_parameters": 60506624,
    }
}

t5_base = {
    "description": "This is the base T5 model by Google.",
    "tasks": ["text-generation"],
    "init_kwargs": {
        "model_path": "t5-base"
    },
    "details": {
        "num_parameters": 222903552,
    }
}


models = {
    "t5-small": t5_small,
    "t5-base": t5_base
}