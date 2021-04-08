models = {
    "bart-large-mnli": {
        "description": "Facebook's large version of BART, finetuned on the Multi-Genre Natural Language Inference dataset.",
        "tasks": ["text-classification"],
        "init_kwargs": {
            "model_path": "facebook/bart-large-mnli"
        },
        "details": {
            "text-classification": {
                "languages": ["eng"],
                "description": "Performs zero-shot classification.",
                "required_params": ["labels"],
                "supported_params": ["allow_multiple"],
                "speed": {
                    "score": 2,
                    "description": "Many parameters"
                },
                "accuracy": {
                    "score": 4,
                    "description": "Score of xx on the xx dataset."
                }
            }
        }
    }
}