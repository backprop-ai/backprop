t5_base_qa_summary_emotion = {
    "description": "This is the T5 base model by Google, and has been finetuned further for Q&A, Summarisation, and Sentiment analysis (emotion detection).",
    "tasks": ["text-generation", "emotion", "summarisation", "qa"],
    "init_kwargs": {
        "model_path": "kiri-ai/t5-base-qa-summary-emotion"
    },
    "details": {
        "num_parameters": 222903552,
        "text-generation": {
            "finetunable": True
        }
    }
}

models = {
    "t5-base-qa-summary-emotion": t5_base_qa_summary_emotion
}