clip = {
    "description": "ViT-B/32 version of OpenAI's CLIP. Combines general knowledge of image and text for a variety of zero-shot applications.",
    "tasks": ["image-classification", "image-vectorisation", "text-vectorisation", "image-text-vectorisation"],
    "init_kwargs": {
        "model_path": "ViT-B/32"
    },
    "details": {
        "num_parameters": 151277313,
        "max_text_length": 77,
        "image-classification": {
            "languages": ["eng"],
            "zero_shot": True,
            "description": "Predicts the most probable label from the ones provided. Returned probabilities always sum to 100%.",
            "score": {
                "value": 5,
                "description": "Accuracy of 80.5% on CIFAR100, 76.1% on ImageNet. High scores on a large variety of benchmarks, including adversarial examples."
            }
        },
        "image-vectorisation": {
            "finetunable": True,
            "vector_size": 512
        },
        "text-vectorisation": {
            "languages": ["eng"],
            "finetunable": True,
            "vector_size": 512,
        },
        "image-text-vectorisation": {
            "languages": ["eng"],
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

models = {
    "clip": clip,
}