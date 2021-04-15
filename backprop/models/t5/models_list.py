t5_small = {
    "description": "Small version of the T5 model by Google. This is a text generation model that can be finetuned to solve virtually any text based task.",
    "tasks": ["text-generation", "summarisation"],
    "init_kwargs": {
        "model_path": "t5-small"
    },
    "details": {
        "num_parameters": 60506624,
        "max_text_length": 512,
        "text-generation": {
            "languages": ["eng"],
            "description": "Summarise text with `summarize: some text`. Translate English to German, French, and Romanian with `translate English to German: Some sentence.`, `translate English to French: Some sentence.`, and `translate English to Romanian: Some sentence.`.",
            "finetunable": True
        },
        "credits": [
            {
                "name": "Google",
                "url": "https://arxiv.org/abs/1910.10683"
            },
            {
                "name": "Hugging Face",
                "url": "https://huggingface.co/transformers/model_doc/t5.html"
            }
        ]
    }
}

t5_base = {
    "description": "Base version of the T5 model by Google. This is a text generation model that can be finetuned to solve virtually any text based task.",
    "tasks": ["text-generation", "summarisation"],
    "init_kwargs": {
        "model_path": "t5-base"
    },
    "details": {
        "num_parameters": 222903552,
        "max_text_length": 512,
        "text-generation": {
            "languages": ["eng"],
            "description": "Summarise text with `summarize: some text`. Translate English to German, French, and Romanian with `translate English to German: Some sentence.`, `translate English to French: Some sentence.`, and `translate English to Romanian: Some sentence.`.",
            "finetunable": True
        },
        "credits": [
            {
                "name": "Google",
                "url": "https://arxiv.org/abs/1910.10683"
            },
            {
                "name": "Hugging Face",
                "url": "https://huggingface.co/transformers/model_doc/t5.html"
            }
        ]
    }
}


models = {
    "t5-small": t5_small,
    "t5-base": t5_base
}