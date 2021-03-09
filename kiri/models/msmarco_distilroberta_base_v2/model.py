from kiri.models import TextVectorisationModel

class MSMARCODistilrobertaBaseV2(TextVectorisationModel):
    def __init__(self, *args, **kwargs):
        TextVectorisationModel.__init__(self, "msmarco-distilroberta-base-v2",
                                *args, **kwargs)

        self.tasks = ["text-vectorisation"]
        self.description = "This English model is a standard distilroberta-base model from the Sentence Transformers repo, which has been trained on the MS MARCO dataset."
        self.name = "msmarco-distilroberta-base-v2"

    def __call__(self, task_input, task="text-vectorisation"):
        if task in ["text-vectorisation", "vectorisation"]:
            text = task_input.pop("text")

            return self.vectorise(text).tolist()
        else:
            raise ValueError(f"Unsupported task: {task}")