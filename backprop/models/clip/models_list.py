clip = {
    "description": "OpenAI's recently released CLIP model — when supplied with a list of labels and an image, CLIP can accurately predict which labels best fit the provided image.",
    "tasks": ["image-classification", "image-vectorisation", "text-vectorisation", "image-text-vectorisation"],
    "init_kwargs": {
        "model_path": "ViT-B/32"
    },
    "details": {
        "num_parameters": 407344131,
    }
}

# TODO: RN50 etc
models = {
    "clip": clip,
    "clip-vit-b32": clip,
}