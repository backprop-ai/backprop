from backprop.models import TextVectorisationModel

class DistiluseBaseMultilingualCasedV2(TextVectorisationModel):
    def __init__(self, *args, **kwargs):
        TextVectorisationModel.__init__(self, "distiluse-base-multilingual-cased-v2",
                                *args, **kwargs)

        self.tasks = ["text-vectorisation"]
        self.description = "This model is based off Sentence-Transformer's distiluse-base-multilingual-cased multilingual model that has been extended to understand sentence embeddings in 50+ languages."
        self.name = "distiluse-base-multilingual-cased-v2"