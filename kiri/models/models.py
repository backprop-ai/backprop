from transformers import AutoModelForPreTraining, AutoTokenizer
from sentence_transformers import SentenceTransformer

class BaseModel:
    def __init__(self, model):
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class PathModel:
    def __init__(self, model_path, tokenizer_path=None, init_model=None,
                init_tokenizer=None, init=True):
        self.initialised = False
        self.init_model = init_model
        self.init_tokenizer = init_tokenizer
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        # Initialise
        if init:
            self.model = self.init_model(model_path)

            # Not all models need tokenizers
            if tokenizer_path:
                self.tokenizer = self.init_tokenizer(tokenizer_path)
            
            self.initialised = True


    def check_init(self):
        # Initialise on call time
        if not self.initialised:
            self.__init__(self.model_path, self.tokenizer_path,
                        init_model=self.init_model, init_tokenizer=self.init_tokenizer,
                        init=True)


    def __call__(self, *args, **kwargs):
        self.check_init()

        return self.model(*args, **kwargs)


class HuggingModel(PathModel):
    def __init__(self, model_path, tokenizer_path=None,
                model_class=AutoModelForPreTraining,
                tokenizer_class=AutoTokenizer, init=True):
        # Usually the same
        if not tokenizer_path:
            tokenizer_path = model_path

        return super().__init__(model_path, tokenizer_path=tokenizer_path,
                                init_model=model_class.from_pretrained,
                                init_tokenizer=tokenizer_class.from_pretrained,
                                init=init)


class VectorisationModel(PathModel):
    def __init__(self, path, model_class=SentenceTransformer, init=True):
        return super().__init__(path, model_class, init=init)


    def __call__(self, *args, **kwargs):
        self.check_init()
        return self.model.encode(*args, **kwargs)


class GenerationModel(HuggingModel):
    def generate(self, text):
        self.check_init()
        features = self.tokenizer(text, return_tensors="pt")

        tokens = self.model.generate(**features)[0]
        output = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return output
