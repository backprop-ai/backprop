efficientnet_description = "EfficientNet is a very efficient image-classification model. Trained on ImageNet."


efficientnet_b0 = {
    "description": efficientnet_description,
    "tasks": ["image-classification"],
    "init_kwargs": {
        "model_path": "efficientnet-b0"
    },
    "details": {
        "num_parameters": 5288548,
        "image-classification": {
            "finetunable": True
        }
    }
}

efficientnet_b4 = {
    "description": efficientnet_description,
    "tasks": ["image-classification"],
    "init_kwargs": {
        "model_path": "efficientnet-b4"
    },
    "details": {
        "num_parameters": 19341616,
        "image-classification": {
            "finetunable": True
        }
    }
}

efficientnet_b7 = {
    "description": efficientnet_description,
    "tasks": ["image-classification"],
    "init_kwargs": {
        "model_path": "efficientnet-b7"
    },
    "details": {
        "num_parameters": 66347960,
        "image-classification": {
            "finetunable": True
        }
    }
}


models = {
    "efficientnet-b0": efficientnet_b0,
    "efficientnet-b4": efficientnet_b4,
    "efficientnet-b7": efficientnet_b7,
}