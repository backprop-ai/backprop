msmarco_distilroberta_base_v2 = {
    "description": "This English model is a standard distilroberta-base model from the Sentence Transformers repo, which has been trained on the MS MARCO dataset.",
    "tasks": ["text-vectorisation"],
    "init_kwargs": {
        "model_path": "msmarco-distilroberta-base-v2",
        "max_length": 512
    },
    "details": {
        "num_parameters": 559893507,
        # "text-vectorisation": {
        #     "languages": ["eng"],
        #     "description": "Performs zero-shot classification.",
        #     "speed": {
        #         "score": 2,
        #         "description": "Many parameters"
        #     },
        #     "accuracy": {
        #         "score": 4,
        #         "description": "Score of xx on the xx dataset."
        #     }
        # },
        # "credits": [
        #     {
        #         "name": "Facebook AI",
        #         "url": ""
        #     },
        #     {
        #         "name": "Hugging Face",
        #         "url": ""
        #     }
        # ]
    }
}

distiluse_base_multilingual_cased_v2 = {
    "description": "This model is based off Sentence-Transformer's distiluse-base-multilingual-cased multilingual model that has been extended to understand sentence embeddings in 50+ languages.",
    "tasks": ["text-vectorisation"],
    "init_kwargs": {
        "model_path": "distiluse-base-multilingual-cased-v2",
        "max_length": 512
    },
    "details": {
        "num_parameters": 559893507,
    }
}



models = {
    "msmarco-distilroberta-base-v2": msmarco_distilroberta_base_v2,
    "distiluse-base-multilingual-cased-v2": distiluse_base_multilingual_cased_v2
}