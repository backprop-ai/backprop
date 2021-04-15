t5_base_qa_summary_emotion = {
    "description": "Base version of Google's T5, finetuned further for Q&A, Summarisation, and Sentiment analysis (emotion detection).",
    "tasks": ["text-generation", "emotion", "summarisation", "qa"],
    "init_kwargs": {
        "model_path": "kiri-ai/t5-base-qa-summary-emotion"
    },
    "details": {
        "num_parameters": 222903552,
        "text-generation": {
            "languages": ["eng"],
            "finetunable": True,
            "description": "While this model supports text generation, it is optimised to solve concrete tasks."
        },
        "emotion": {
            "languages": ["eng"],
            "finetunable": True,
            "description": "The response is a string of comma separated emotions. The emotions are from: neutral, admiration, approval, annoyance, gratitude, disapproval, amusement, curiosity, love, optimism, disappointment, joy, realization, anger, sadness, confusion, caring, excitement, surprise, disgust, desire, fear, remorse, embarrassment, nervousness, pride, relief, grief."
        },
        "summarisation": {
            "languages": ["eng"],
            "finetunable": True
        },
        "qa": {
            "languages": ["eng"],
            "conversational": True,
            "finetunable": True,
            "description": "Trained for conversational question answering - supports previous question-answer pairs.",
            "score": {
                "value": 4,
                "description": "F1 79.5 on SQuAD 2.0, F1 70.6 on CoQa dev sets."
            }
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
    "t5-base-qa-summary-emotion": t5_base_qa_summary_emotion
}