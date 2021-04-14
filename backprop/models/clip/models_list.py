clip_vit_b32 = {
    "description": "OpenAI's recently released CLIP model — when supplied with a list of labels and an image, CLIP can accurately predict which labels best fit the provided image.",
    "tasks": ["image-classification", "image-vectorisation", "text-vectorisation", "image-text-vectorisation"],
    "init_kwargs": {
        "model_path": "ViT-B/32"
    },
    "details": {
        "num_parameters": 151277313,
        "max_text_length": 77,
        "image-classification": {
            "zero_shot": True
        },
        "image-vectorisation": {
            "finetunable": True,
            "vector_size": 512
        },
        "text-vectorisation": {
            "finetunable": True,
            "vector_size": 512,
        },
        "image-text-vectorisation": {
            "finetunable": True,
            "vector_size": 1024,
        },
        "credits": [
            {
                "name": "OpenAI",
                "url": "https://openai.com/blog/clip/"
            }
        ]
    }
}

# TODO: RN50 etc
models = {
    "clip-vit-b32": clip_vit_b32,
}