from typing import List, Tuple
from transformers import AutoModelForPreTraining, AutoTokenizer, \
    AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Subset
from sentence_transformers import SentenceTransformer
from functools import partial
import os

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.memory import garbage_collection_cuda

class BaseModel(torch.nn.Module):
    """
    The base class for a model.
    
    Attributes:
        model: Your model that takes some args, kwargs and returns an output.
            Must be callable.
    """
    def __init__(self, model, name: str = None, description: str = None, tasks: List[str] = None):
        torch.nn.Module.__init__(self)
        self.model = model
        self.name = name or "base-model"
        self.description = description or "This is the base description. Change me."
        self.tasks = tasks or [] # Supports no tasks

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def finetune(self, *args, **kwargs):
        raise NotImplementedError("This model does not support finetuning")

    def to(self, device):
        self.model.to(device)
        self._model_device = device
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    # def set_name(self, name: str):


class Finetunable(pl.LightningModule):
    """
    Makes a model easily finetunable.
    """
    def __init__(self):
        pl.LightningModule.__init__(self)

        self.batch_size = 1

    def finetune(self, dataset, validation_split: float = 0.15, train_idx: List[int] = None,
                val_idx: List[int] = None, epochs: int = 20, batch_size: int = None,
                optimal_batch_size: int = None, early_stopping: bool = True, trainer = None):
        self.batch_size = batch_size or 1

        if not torch.cuda.is_available():
            raise Exception("You need a cuda capable (Nvidia) GPU for finetuning")
        
        dataset_train = None
        dataset_valid = None

        if train_idx and val_idx:
            dataset_train = Subset(dataset, train_idx)
            dataset_valid = Subset(dataset, val_idx)
        else:
            len_train = int(len(dataset) * (1 - validation_split))
            len_valid = len(dataset) - len_train
            dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [len_train, len_valid])

        if not hasattr(self, "dataset_train") and not hasattr(self, "dataset_valid"):
            self.dataset_train = dataset_train
            self.dataset_valid = dataset_valid

        if batch_size == None:
            # Find batch size
            temp_trainer = pl.Trainer(auto_scale_batch_size="power", gpus=-1)
            print("Finding the optimal batch size...")
            temp_trainer.tune(self)

            # Ensure that memory gets cleared
            del self.trainer
            del temp_trainer
            garbage_collection_cuda()

        trainer_kwargs = {}
        
        if optimal_batch_size:
            # Don't go over
            batch_size = min(self.batch_size, optimal_batch_size)
            accumulate_grad_batches = max(1, int(optimal_batch_size / batch_size))
            trainer_kwargs["accumulate_grad_batches"] = accumulate_grad_batches
        
        if early_stopping:
            # Stop when val loss stops improving
            early_stopping = EarlyStopping(monitor="val_loss", patience=1)
            trainer_kwargs["callbacks"] = [early_stopping]

        if not trainer:
            trainer = pl.Trainer(gpus=-1, max_epochs=epochs, checkpoint_callback=False,
                logger=False, **trainer_kwargs)

        self.model.train()
        trainer.fit(self)

        del self.dataset_train
        del self.dataset_valid
        del self.trainer

        # For some reason the model can end up on CPU after training
        self.to(self._model_device)
        self.model.eval()
        print("Training finished! Save your model for later with backprop.save or upload it with backprop.upload")

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() or 0,
            sampler=self.dl_sampler(self.dataset_train) or None)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() or 0,
            sampler=self.dl_sampler(self.dataset_valid) or None)
    
    def configure_optimizers(self):
        raise NotImplementedError("configure_optimizers must be implemented")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("training_step must be implemented")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("validation_step must be implemented")



class PathModel(BaseModel):
    """
    Class for models which are initialised from a path.

    Attributes:
        model_path: Path to the model
        init_model: Callable to initialise model from path
        tokenizer_path (optional): Path to the tokenizer
        init_tokenizer (optional): Callable to initialise tokenizer from path
        device (optional): Device for inference. Defaults to "cuda" if available.
    """
    def __init__(self, model_path, init_model, tokenizer_path=None,
                init_tokenizer=None, device=None):
        BaseModel.__init__(self, None)
        self.init_model = init_model
        self.init_tokenizer = init_tokenizer
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self._model_device = device

        if self._model_device is None:
            self._model_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialise
        self.model = self.init_model(model_path).eval().to(self._model_device)

        # Not all models need tokenizers
        if self.tokenizer_path:
            self.tokenizer = self.init_tokenizer(self.tokenizer_path)
        

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class HuggingModel(PathModel):
    """
    Class for models which are initialised from a local path or huggingface

    Attributes:
        model_path: Local or huggingface.co path to the model
        init_model: Callable to initialise model from path
            Defaults to AutoModelForPreTraining from huggingface
        tokenizer_path (optional): Path to the tokenizer
        init_tokenizer (optional): Callable to initialise tokenizer from path
            Defaults to AutoTokenizer from huggingface.
        device (optional): Device for inference. Defaults to "cuda" if available.
    """
    def __init__(self, model_path, tokenizer_path=None,
                model_class=AutoModelForPreTraining,
                tokenizer_class=AutoTokenizer, device=None):
        # Usually the same
        if not tokenizer_path:
            tokenizer_path = model_path

        # Object was made with init = False
        if hasattr(self, "initialised"):
            model_path = self.model_path
            tokenizer_path = self.tokenizer_path
            init_model = self.init_model
            init_tokenizer = self.init_tokenizer
            device = self._model_device
        else:
            init_model = model_class.from_pretrained
            init_tokenizer = tokenizer_class.from_pretrained

        return PathModel.__init__(self, model_path, tokenizer_path=tokenizer_path,
                                init_model=init_model,
                                init_tokenizer=init_tokenizer,
                                device=device)


class TextVectorisationModel(PathModel):
    """
    Class for models which are initialised from a local path or Sentence Transformers

    Attributes:
        model_path: Local, sentence transformers or huggingface.co path to the model
        init_model: Callable to initialise model from path
            Defaults to SentenceTransformer
        device (optional): Device for inference. Defaults to "cuda" if available.
    """
    def __init__(self, model_path, init_model=SentenceTransformer,
                device=None):
        init_model = partial(init_model, device=device)

        PathModel.__init__(self, model_path,
                                init_model=init_model,
                                device=device)

        self.tasks = ["text-vectorisation"]
        self.description = "This is a text vectorisation model"
        self.name = "text-vec-model"
        self.max_length = 512

    def __call__(self, task_input, task="text-vectorisation", return_tensor=False, preprocess=True, train=False):
        is_list = False
        if task == "text-vectorisation":
            input_ids = None
            attention_mask = None

            if preprocess:
                text = task_input.get("text")

                if type(text) == list:
                    is_list = True
                else:
                    text = [text]

                features = self.process_text(text).to(self._model_device)
            else:
                features = task_input.get("text")
                is_list = True

            with torch.set_grad_enabled(train):
                text_vecs = self.vectorise(features)

            if not return_tensor:
                text_vecs = text_vecs.tolist()

            output = text_vecs

            if not is_list:
                output = output[0]

            return output
        else:
            raise ValueError(f"Unsupported task '{task}'")

    def process_text(self, input_text, max_length=None, padding=True):
        max_length = max_length or self.max_length

        if max_length > self.max_length:
            raise ValueError(f"This model has a max_length limit of {self.max_length}")

        processed = self.model.tokenizer(input_text, truncation=True, padding=padding,
                    return_tensors="pt", max_length=max_length)

        return processed

    def vectorise(self, features):
        return self.model.forward(features)["sentence_embedding"]

    def configure_optimizers(self):
        return AdamW(params=self.model.parameters(), lr=2e-5, eps=1e-6, correct_bias=False)


class TextGenerationModel(HuggingModel):
    """
    Class for models which are initialised from a local path or Huggingface

    Attributes:
        *args and **kwargs are passed to HuggingModel's __init__
    """
    def generate(self, text, **kwargs):
        """
        Generate according to the model's generate method.
        """
        # Get and remove do_sample or set to False
        do_sample = kwargs.pop("do_sample", None) or False
        params = ["temperature", "top_k", "top_p", "repetition_penalty",
                    "length_penalty", "num_beams", "num_return_sequences", "num_generations"]

        # If params are changed, we want to sample
        for param in params:
            if param in kwargs.keys() and kwargs[param] != None:
                do_sample = True
                break

        if "temperature" in kwargs:
            # No sampling
            if kwargs["temperature"] == 0.0:
                do_sample = False
                del kwargs["temperature"]

        # Override, name correctly
        if "num_generations" in kwargs:
            if kwargs["num_generations"] != None:
                kwargs["num_return_sequences"] = kwargs["num_generations"]
                
            del kwargs["num_generations"]

        is_list = False
        if isinstance(text, list):
            is_list = True

        if not is_list:
            text = [text]

        all_tokens = []
        for text in text:
            features = self.tokenizer(text, return_tensors="pt")

            for k, v in features.items():
                features[k] = v.to(self._model_device)

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
    """
    Class for classification models which are initialised from a local path or huggingface

    Attributes:
        model_path: Local or huggingface.co path to the model
        tokenizer_path (optional): Path to the tokenizer
        model_class (optional): Callable to initialise model from path
            Defaults to AutoModelForSequenceClassification from huggingface
        tokenizer_class (optional): Callable to initialise tokenizer from path
            Defaults to AutoTokenizer from huggingface.
        device (optional): Device for inference. Defaults to "cuda" if available.
    """
    def __init__(self, model_path, tokenizer_path=None,
                model_class=AutoModelForSequenceClassification,
                tokenizer_class=AutoTokenizer, device=None):
        return super().__init__(model_path, tokenizer_path=tokenizer_path,
                    model_class=model_class, tokenizer_class=tokenizer_class,
                    device=device)

    def calculate_probability(self, text, label, device):
        hypothesis = f"This example is {label}."
        features = self.tokenizer.encode(text, hypothesis, return_tensors="pt",
                                    truncation=True).to(self._model_device)
        logits = self.model(features)[0]
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:, 1]
        return prob_label_is_true.item()


    def classify(self, text, labels):
        """
        Classifies text, given a set of labels.
        """
        if isinstance(text, list):
            # Must have a consistent amount of examples
            assert(len(text) == len(labels))
            # TODO: implement proper batching
            results_list = []
            for text, labels in zip(text, labels):
                results = {}
                for label in labels:
                    results[label] = self.calculate_probability(text, label, self._model_device)

                results_list.append(results)

            return results_list
        else:
            results = {}
            for label in labels:
                results[label] = self.calculate_probability(
                    text, label, self._model_device)

            return results