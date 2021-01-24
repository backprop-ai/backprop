from transformers import AutoModelForPreTraining, AutoTokenizer, \
    AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch

class BaseModel:
    def __init__(self, model):
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class PathModel(BaseModel):
    def __init__(self, model_path, tokenizer_path=None, init_model=None,
                init_tokenizer=None, device=None, init=True):
        self.initialised = False
        self.init_model = init_model
        self.init_tokenizer = init_tokenizer
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = device

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialise
        if init:
            self.model = self.init_model(model_path).eval().to(self.device)

            # Not all models need tokenizers
            if self.tokenizer_path:
                self.tokenizer = self.init_tokenizer(self.tokenizer_path)
            
            self.initialised = True


    def check_init(self):
        # Initialise on call time
        if not self.initialised:
            self.__init__(self.model_path, init=True)


    def __call__(self, *args, **kwargs):
        self.check_init()

        return self.model(*args, **kwargs)


class HuggingModel(PathModel):
    def __init__(self, model_path, tokenizer_path=None,
                model_class=AutoModelForPreTraining,
                tokenizer_class=AutoTokenizer, device=None, init=True):
        # Usually the same
        if not tokenizer_path:
            tokenizer_path = model_path

        # Object was made with init = False
        if hasattr(self, "initialised"):
            model_path = self.model_path
            tokenizer_path = self.tokenizer_path
            init_model = self.init_model
            init_tokenizer = self.init_tokenizer
            device = self.device
        else:
            init_model = model_class.from_pretrained
            init_tokenizer = tokenizer_class.from_pretrained

        return super().__init__(model_path, tokenizer_path=tokenizer_path,
                                init_model=init_model,
                                init_tokenizer=init_tokenizer,
                                device=device, init=init)


class VectorisationModel(PathModel):
    def __init__(self, model_path, model_class=SentenceTransformer,
                device=None, init=True):
        # Object was made with init = False
        if hasattr(self, "initialised"):
            model_path = self.model_path
            init_model = self.init_model
            device = self.device
        else:
            init_model = model_class

        return super().__init__(model_path,
                                init_model=init_model,
                                device=device, init=init)


    def __call__(self, *args, **kwargs):
        return self.vectorise(*args, **kwargs)

    def vectorise(self, *args, **kwargs):
        self.check_init()
        with torch.no_grad():
            return self.model.encode(*args, **kwargs)


class GenerationModel(HuggingModel):
    def generate(self, text, **kwargs):
        self.check_init()

        # Get and remove do_sample or set to False
        do_sample = kwargs.pop("do_sample", None) or False
        params = ["temperature", "top_k", "top_p", "repetition_penalty",
                    "length_penalty", "num_beams", "num_return_sequences"]

        # If params are changed, we want to sample
        for param in params:
            if param in kwargs.keys():
                do_sample = True
                break

        is_list = False
        if isinstance(text, list):
            is_list = True

        if not is_list:
            text = [text]

        all_tokens = []
        for text in text:
            features = self.tokenizer(text, return_tensors="pt")

            for k, v in features.items():
                features[k] = v.to(self.device)

            with torch.no_grad():
                tokens = self.model.generate(do_sample=do_sample,
                                            **features, **kwargs)

            all_tokens.append(tokens)
            
        value = []
        for tokens in all_tokens:
            value.append([self.tokenizer.decode(tokens, skip_special_tokens=True)
                    for tokens in tokens])
        
        output = value

        # Unwrap generation list
        if kwargs.get("num_return_sequences", 1) == 1:
            output_unwrapped = []
            for value in output:
                output_unwrapped.append(value[0])

            output = output_unwrapped
        
        # Return single item
        if not is_list:
            output = output[0]

        return output


class ClassificationModel(HuggingModel):
    def __init__(self, model_path, tokenizer_path=None,
                model_class=AutoModelForSequenceClassification,
                tokenizer_class=AutoTokenizer, device=None, init=True):
        return super().__init__(model_path, tokenizer_path=tokenizer_path,
                    model_class=model_class, tokenizer_class=tokenizer_class,
                    device=device, init=init)

    def calculate_probability(self, text, label, device):
        hypothesis = f"This example is {label}."
        features = self.tokenizer.encode(text, hypothesis, return_tensors="pt",
                                    truncation=True).to(self.device)
        logits = self.model(features)[0]
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:, 1]
        return prob_label_is_true.item()


    def classify(self, text, labels):
        self.check_init()
        if isinstance(text, list):
            # Must have a consistent amount of examples
            assert(len(text) == len(labels))
            # TODO: implement proper batching
            results_list = []
            for text, labels in zip(text, labels):
                results = {}
                for label in labels:
                    results[label] = self.calculate_probability(text, label, self.device)

                results_list.append(results)

            return results_list
        else:
            results = {}
            for label in labels:
                results[label] = self.calculate_probability(
                    text, label, self.device)

            return results