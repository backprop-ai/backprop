gpt2_large = {
    "description": "Large version of OpenAI's GPT-2 model. This is a general-use model, and has not been further finetuned on any specific languages or tasks.",
    "tasks": ["text-generation"],
    "init_kwargs": {
        "model_path": "gpt2-large"
    },
    "details": {
        "num_parameters": 774030080,
        "max_text_length": 1024,
        "text-generation": {
            "languages": ["eng"],
            "description": "As the model was trained on unfiltered content from the internet, be vary of biases and 'facts' that sound true."
        },
        "credits": [
            {
                "name": "OpenAI",
                "url": "https://openai.com/blog/better-language-models/"
            },
            {
                "name": "Hugging Face",
                "url": "https://huggingface.co/transformers/model_doc/gpt2.html"
            }
        ]
    }
}

gpt2_medium = {
    "description": "Medium version of OpenAI's GPT-2 model. This is a general-use model, and has not been further finetuned on any specific languages or tasks.",
    "tasks": ["text-generation"],
    "init_kwargs": {
        "model_path": "gpt2-medium"
    },
    "details": {
        "num_parameters": 354823168,
        "max_text_length": 1024,
        "text-generation": {
            "languages": ["eng"],
            "description": "As the model was trained on unfiltered content from the internet, be vary of biases and 'facts' that sound true. Default temperature as 0.7."
        },
        "credits": [
            {
                "name": "OpenAI",
                "url": "https://openai.com/blog/better-language-models/"
            },
            {
                "name": "Hugging Face",
                "url": "https://huggingface.co/transformers/model_doc/gpt2.html"
            }
        ]
    }
}

distilgpt2 = {
    "description": "Distilled version of OpenAI's GPT-2 model. This is a general-use model, and has not been further finetuned on any specific languages or tasks.",
    "tasks": ["text-generation"],
    "init_kwargs": {
        "model_path": "distilgpt2"
    },
    "details": {
        "num_parameters": 81912576,
        "max_text_length": 1024,
        "text-generation": {
            "languages": ["eng"],
            "description": "As the model was trained on unfiltered content from the internet, be vary of biases and 'facts' that sound true. Default temperature as 0.7."
        },
        "credits": [
            {
                "name": "OpenAI",
                "url": "https://openai.com/blog/better-language-models/"
            },
            {
                "name": "Hugging Face",
                "url": "https://huggingface.co/transformers/model_doc/gpt2.html"
            }
        ]
    }
}


models = {
    "gpt2-large": gpt2_large,
    "gpt2-medium": gpt2_medium,
    "distilgpt2": distilgpt2
}