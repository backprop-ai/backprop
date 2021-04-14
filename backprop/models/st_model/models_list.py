msmarco_distilroberta_base_v2 = {
    "description": "A distilled version of Facebook AI's Base RoBERTa Model. It has been finetuned on the MS MARCO dataset to produce accurate vectors for semantic search.",
    "tasks": ["text-vectorisation"],
    "init_kwargs": {
        "model_path": "msmarco-distilroberta-base-v2",
        "max_length": 512
    },
    "max_text_length": 512,
    "details": {
        "num_parameters": 82118400,
        "text-vectorisation": {
            "languages": ["eng"],
            "description": "The vectors are optimised to maximise similarity between a paragraph of text and query, making it suited for search/qa related applications in English.",
            "finetunable": True,
            "vector_size": 768
        },
        "credits": [
            {
                "name": "Facebook AI",
                "url": "https://arxiv.org/abs/1907.11692"
            },
            {
                "name": "Sentence Transformers",
                "url": "https://github.com/UKPLab/sentence-transformers"
            }
        ]
    }
}

distiluse_base_multilingual_cased_v2 = {
    "description": "Multilingual knowledge distilled version of Google's multilingual Universal Sentence Encoder. This version has been extended to understand 50+ languages.",
    "tasks": ["text-vectorisation"],
    "init_kwargs": {
        "model_path": "distiluse-base-multilingual-cased-v2",
        "max_length": 512
    },
    "details": {
        "num_parameters": 135127808,
        "max_text_length": 512,
        "text-vectorisation": {
            "languages": ["ara", "bul", "cat", "ces", "dan", "deu", "ell", "spa", "est", "fas", "fin", "fra", "glg", "guj", "heb", "hin", "hrv", "hun", "hye", "ind", "ita", "jpn", "kat", "kor", "kur", "lit", "lav", "mkd", "mon", "mar", "msa", "mya", "nob", "nld", "pol", "por", "ron", "rus", "slk", "slv", "sqi", "srp", "swe", "tha", "tur", "ukr", "urd", "vie", "zho"],
            "description": "While the model has been trained to produce vectors for sentences, it can encode paragraphs of text as well.",
            "finetunable": True,
            "vector_size": 512
        },
        "credits": [
            {
                "name": "Google",
                "url": "https://arxiv.org/abs/1907.04307"
            },
            {
                "name": "Sentence Transformers",
                "url": "https://github.com/UKPLab/sentence-transformers"
            }
        ]
    }
}



models = {
    "msmarco-distilroberta-base-v2": msmarco_distilroberta_base_v2,
    "distiluse-base-multilingual-cased-v2": distiluse_base_multilingual_cased_v2
}