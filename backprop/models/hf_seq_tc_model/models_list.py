xlnet_base_cased = {
    "description": "XLNet",
    "tasks": ["text-classification"],
    "init_kwargs": {
        "model_path": "xlnet-base-cased"
    }
}

models = {
    "xlnet-base-cased": xlnet_base_cased
}