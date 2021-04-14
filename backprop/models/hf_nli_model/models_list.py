bart_large_mnli = {
    "description": "Large version of Facebook AI's BART, finetuned on the Multi-Genre Natural Language Inference dataset for zero-shot text classification.",
    "tasks": ["text-classification"],
    "init_kwargs": {
        "model_path": "facebook/bart-large-mnli"
    },
    "details": {
        "num_parameters": 407344131,
        "max_text_length": 1024,
        "text-classification": {
            "languages": ["eng"],
            "zero_shot": True,
            "score": {
                "value": 4,
                "description": "Matched accuracy of 89.9% and mismatched accuracy of 90.01% on the MNLI test set."
            }
        },
        "credits": [
            {
                "name": "Facebook AI",
                "url": "https://arxiv.org/abs/1910.13461"
            },
            {
                "name": "Hugging Face",
                "url": "https://huggingface.co/transformers/model_doc/bart.html"
            }
        ]
    }
}


deberta_base_mnli = {
    "description": "Base version of Microsoft's DeBERTa, finetuned on the Multi-Genre Natural Language Inference dataset for zero-shot text classification.",
    "tasks": ["text-classification"],
    "init_kwargs": {
        "model_path": "microsoft/deberta-base-mnli"
    },
    "details": {
        "num_parameters": 139194627,
        "max_text_length": 512,
        "text-classification": {
            "languages": ["eng"],
            "zero_shot": True,
            "score": {
                "value": 4,
                "description": "Matched accuracy of 88.8% on the MNLI test set."
            }
        },
        "credits": [
            {
                "name": "Microsoft",
                "url": "https://arxiv.org/abs/2006.03654"
            },
            {
                "name": "Hugging Face",
                "url": "https://huggingface.co/transformers/model_doc/deberta.html"
            }
        ]
    }
}

xlmr_large_xnli = {
    "description": "XLM-RoBERTa is a multilingual variant of Facebook's RoBERTa model. It has been finetuned on the XNLI dataset for multilingual zero-shot text classification.",
    "tasks": ["text-classification"],
    "init_kwargs": {
        "model_path": "joeddav/xlm-roberta-large-xnli"
    },
    "details": {
        "num_parameters": 559893507,
        "max_input_text_length": 512,
        "text-classification": {
            "languages": ["eng", "deu", "fra", "spa", "ell", "bul", "rus", "tur", "ara", "vie", "tha", "zho", "hin", "swa", "urd"],
            "description": "The model has been explicitly finetuned on 15 languages, while the base model was trained on 100 different languages. This makes the model suited for multilingual zero-shot classification scenarios.",
            "zero_shot": True,
            # "score": {
            #     "value": 4,
            #     "description": "Score of xx on the xx dataset."
            # }
        },
        "credits": [
            {
                "name": "Facebook AI",
                "url": "https://arxiv.org/abs/1911.02116"
            },
            {
                "name": "Joe Davison @ Hugging Face",
                "url": "https://huggingface.co/joeddav/xlm-roberta-large-xnli"
            }
        ]
    }
}

models = {
    "bart-large-mnli": bart_large_mnli,
    "xlmr-large-xnli": xlmr_large_xnli,
    "deberta-base-mnli": deberta_base_mnli
}