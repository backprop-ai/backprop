efficientnet_b0 = {
    "description": "B0 (small) version of EfficientNet. Trained on ImageNet's 1000 classes for image classification.",
    "tasks": ["image-classification"],
    "init_kwargs": {
        "model_path": "efficientnet-b0"
    },
    "details": {
        "num_parameters": 5288548,
        "image-classification": {
            "finetunable": True,
            "score": {
                "value": 3,
                "description": "Top 1 accuracy of 76.3% on the ImageNet test set."
            }
        },
        "credits": [
            {
                "name": "Google",
                "url": "https://arxiv.org/abs/1905.11946"
            },
            {
                "name": "EfficientNet PyTorch",
                "url": "https://github.com/lukemelas/EfficientNet-PyTorch"
            }
        ]
    }
}

efficientnet_b4 = {
    "description": "B4 (base) version of EfficientNet. Trained on ImageNet's 1000 classes for image classification.",
    "tasks": ["image-classification"],
    "init_kwargs": {
        "model_path": "efficientnet-b4"
    },
    "details": {
        "num_parameters": 19341616,
        "image-classification": {
            "finetunable": True,
            "score": {
                "value": 4,
                "description": "Top 1 accuracy of 82.6% on the ImageNet test set."
            }
        },
        "credits": [
            {
                "name": "Google",
                "url": "https://arxiv.org/abs/1905.11946"
            },
            {
                "name": "EfficientNet PyTorch",
                "url": "https://github.com/lukemelas/EfficientNet-PyTorch"
            }
        ]
    }
}

efficientnet_b7 = {
    "description": "B7 (large) version of EfficientNet. Trained on ImageNet's 1000 classes for image classification.",
    "tasks": ["image-classification"],
    "init_kwargs": {
        "model_path": "efficientnet-b7"
    },
    "details": {
        "num_parameters": 66347960,
        "image-classification": {
            "finetunable": True,
            "score": {
                "value": 4,
                "description": "Top 1 accuracy of 84.4% on the ImageNet test set."
            }
        },
        "credits": [
            {
                "name": "Google",
                "url": "https://arxiv.org/abs/1905.11946"
            },
            {
                "name": "EfficientNet PyTorch",
                "url": "https://github.com/lukemelas/EfficientNet-PyTorch"
            }
        ]
    }
}


models = {
    "efficientnet-b0": efficientnet_b0,
    "efficientnet-b4": efficientnet_b4,
    "efficientnet-b7": efficientnet_b7,
}