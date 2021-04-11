clip_vit_b32 = {
    "description": "OpenAI's recently released CLIP model — when supplied with a list of labels and an image, CLIP can accurately predict which labels best fit the provided image.",
    "tasks": ["image-classification", "image-vectorisation", "text-vectorisation", "image-text-vectorisation"],
    "init_kwargs": {
        "model_path": "ViT-B/32"
    },
    "details": {
        "num_parameters": 151277313,
        "image-vectorisation": {
            "finetunable": True
        },
        "text-vectorisation": {
            "finetunable": True
        },
        "image-text-vectorisation": {
            "finetunable": True
        }
    }
}

# TODO: RN50 etc
models = {
    "clip-vit-b32": clip_vit_b32,
}