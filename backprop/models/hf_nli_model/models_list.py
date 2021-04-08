bart_large_mnli = {
    "description": "Facebook's large version of BART, finetuned on the Multi-Genre Natural Language Inference dataset.",
    "tasks": ["text-classification"],
    "init_kwargs": {
        "model_path": "facebook/bart-large-mnli"
    },
    "details": {
        "num_parameters": 407344131,
        "text-classification": {
            "languages": ["eng"],
            "description": "Performs zero-shot classification.",
            "zero_shot": True,
            "speed": {
                "score": 2,
                "description": "Many parameters"
            },
            "accuracy": {
                "score": 4,
                "description": "Score of xx on the xx dataset."
            }
        },
        "credits": [
            {
                "name": "Facebook AI",
                "url": ""
            },
            {
                "name": "Hugging Face",
                "url": ""
            }
        ]
    }
}

xlmr_large_xnli = {
    "description": "XLM-RoBERTa is a multilingual variant of Facebook's RoBERTa model. This has been finetuned on the XNLI dataset, resulting in classification system that is effective on 100 different languages.",
    "tasks": ["text-classification"],
    "init_kwargs": {
        "model_path": "joeddav/xlm-roberta-large-xnli"
    },
    "details": {
        "num_parameters": 559893507,
        "text-classification": {
            "languages": ["eng"],
            "description": "Performs zero-shot classification.",
            "required_optional_params": ["labels"],
            "supported_optional_params": ["allow_multiple"],
            "speed": {
                "score": 2,
                "description": "Many parameters"
            },
            "accuracy": {
                "score": 4,
                "description": "Score of xx on the xx dataset."
            }
        },
        "credits": [
            {
                "name": "Facebook AI",
                "url": ""
            },
            {
                "name": "Hugging Face",
                "url": ""
            }
        ]
    }
}


models = {
    "bart-large-mnli": bart_large_mnli,
    "xlmr-large-xnli": xlmr_large_xnli
}