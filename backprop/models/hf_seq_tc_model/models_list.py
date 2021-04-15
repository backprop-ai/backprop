xlnet_base_cased = {
    "description": "XLNet",
    "tasks": ["text-classification"],
    "init_kwargs": {
        "model_path": "xlnet-base-cased"
    },
    "details": {
        "library_only": True,
        "num_parameters": 117310466,
        "max_text_length": "no limit",
        "text-classification": {
            "description": "This model needs to be finetuned via the library before it can be used for text classification",
            "finetunable": True
        }
    }
}

models = {
    "xlnet-base-cased": xlnet_base_cased
}