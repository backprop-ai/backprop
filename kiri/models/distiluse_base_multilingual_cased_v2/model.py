from kiri.models import TextVectorisationModel

class DistiluseBaseMultilingualCasedV2(TextVectorisationModel):
    def __init__(self, *args, **kwargs):
        TextVectorisationModel.__init__(self, "distiluse-base-multilingual-cased-v2",
                                *args, **kwargs)

        self.tasks = ["text-vectorisation"]
        self.description = "This model is based off Sentence-Transformer's distiluse-base-multilingual-cased multilingual model that has been extended to understand sentence embeddings in 50+ languages."
        self.name = "distiluse-base-multilingual-cased-v2"

    def __call__(self, task_input, task="text-vectorisation"):
        if task in ["text-vectorisation", "vectorisation"]:
            text = task_input.pop("text")

            return self.vectorise(text).tolist()
        else:
            raise ValueError(f"Unsupported task: {task}")