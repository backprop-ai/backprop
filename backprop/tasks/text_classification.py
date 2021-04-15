from typing import List, Tuple, Union, Optional
from backprop.models import BaseModel, AutoModel
from .base import Task
import torch
from transformers.optimization import AdamW
from backprop.utils.datasets import SingleLabelTextClassificationDataset

import requests

TASK = "text-classification"

DEFAULT_LOCAL_MODEL = "english-base"

LOCAL_ALIASES = {
    "english": "deberta-base-mnli",
    "english-base": "deberta-base-mnli",
    "english-large": "bart-large-mnli",
    "multilingual": "xlmr-large-xnli",
    "multilingual-large": "xlmr-large-xnli"
}

class TextClassification(Task):
    """
    Task for classification.

    Attributes:
        model:
            1. Model name
            2. Model name on Backprop's text-classification endpoint
            3. Model object that implements the text-classification task
        local (optional): Run locally. Defaults to False
        api_key (optional): Backprop API key for non-local inference
        device (optional): Device to run inference on. Defaults to "cuda" if available.
    """
    def __init__(self, model: Union[str, BaseModel] = None,
                local: bool = False, api_key: str = None, device: str = None):
        models = AutoModel.list_models(task=TASK)

        super().__init__(model, local=local, api_key=api_key, device=device,
                        models=models, task=TASK,
                        default_local_model=DEFAULT_LOCAL_MODEL,
                        local_aliases=LOCAL_ALIASES)

    @staticmethod
    def list_models(return_dict=False, display=False, limit=None):
        """
        Returns the list of models that can be used and finetuned with this task.

        Args:
            return_dict: Default False. True if you want to return in dict form. Otherwise returns list form.
            display: Default False. True if you want output printed directly (overrides return_dict, and returns nothing).
            limit: Default None. Maximum number of models to return -- leave None to get all models.
        """
        return AutoModel.list_models(task=TASK, return_dict=return_dict, display=display, limit=limit, aliases=LOCAL_ALIASES)
    
    def __call__(self, text: Union[str, List[str]], labels: Optional[Union[List[str], List[List[str]]]] = None,
                top_k: int = 0):
        """Classify input text based on previous training (user-tuned models) or according to given list of labels (zero-shot)

        Args:
            text: string or list of strings to be classified
            labels: list of labels for zero-shot classification (on our out-of-the-box models).
                    If using a user-trained model (e.g. XLNet), this is not used.
            top_k: return probabilities only for top_k predictions. Use 0 to get all.

        Returns:
            dict where each key is a label and value is probability between 0 and 1, or list of dicts.
        """
        task_input = {
            "text": text,
            "labels": labels,
            "top_k": top_k,
        }

        if top_k == 0:
            task_input.pop("top_k")

        if self.local:               
            return self.model(task_input, task=TASK)
        else:
            task_input["model"] = self.model

            res = requests.post("https://api.backprop.co/text-classification", json=task_input,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            return res["probabilities"]
    

    def step(self, batch, batch_idx):
        """
        Performs a training step and returns loss.

        Args:
            batch: Batch output from the dataloader
            batch_idx: Batch index.
        """
        return self.model.training_step(batch)
    
    def configure_optimizers(self):
        """
        Returns default optimizer for text classification (AdamW, learning rate 2e-5)
        """
        return AdamW(params=self.model.parameters(), lr=2e-5)

    def finetune(self, params, validation_split: Union[float, Tuple[List[int], List[int]]]=0.15,
                 max_length: int=128,
                 epochs: int=20, batch_size: int=None, optimal_batch_size: int=None,
                 early_stopping_epochs: int=1, train_dataloader=None, val_dataloader=None, 
                 step=None, configure_optimizers=None):
        """
        Finetunes a text classification model on provided data.

        Args:
            params: Dict containing keys "texts" and "labels", with values being input/output data lists.
            validation_split: Float between 0 and 1 that determines percentage of data to use for validation.
            max_length: Int determining the maximum token length of input strings.
            epochs: Integer specifying how many training iterations to run.
            batch_size: Batch size when training. Leave as None to automatically determine batch size.
            optimal_batch_size: Optimal batch size for the model being trained -- defaults to model settings.
            early_stopping_epochs: Integer determining how many epochs will run before stopping without an improvement in validation loss.
            train_dataloader: Dataloader for providing training data when finetuning. Defaults to inbuilt dataloder.
            val_dataloader: Dataloader for providing validation data when finetuning. Defaults to inbuilt dataloader.
            step: Function determining how to call model for a training step. Defaults to step defined in this task class.
            configure_optimizers: Function that sets up the optimizer for training. Defaults to optimizer defined in this task class.
        
        Examples::
            
            import backprop

            tc = backprop.TextCLassification()

            # Set up input data. Labels will automatically be used to set up model with number of classes for classification.
            inp = ["This is a political news article", "This is a computer science research paper", "This is a movie review"]
            out = ["Politics", "Science", "Entertainment"]
            params = {"texts": inp, "labels": out}

            # Finetune
            tc.finetune(params)
        """
        inputs = params["texts"]
        outputs = params["labels"]

        assert len(inputs) == len(outputs)

        step = step or self.step
        configure_optimizers = configure_optimizers or self.configure_optimizers

        labels = set(outputs)
        class_to_idx = {v: k for k, v in enumerate(labels)}
        labels = {k: v for k,v in enumerate(labels)}

        output_classes = [class_to_idx[i] for i in outputs]

        if hasattr(self.model, "pre_finetuning"):
            self.model.pre_finetuning(labels)

        dataset_params = {
            "inputs": inputs,
            "labels": outputs,
            "class_to_idx": class_to_idx,
            "max_length": max_length
        }

        dataset = SingleLabelTextClassificationDataset(dataset_params, process_batch=self.model.process_batch, length=len(inputs))

        super().finetune(dataset=dataset, validation_split=validation_split, 
                        epochs=epochs, batch_size=batch_size, optimal_batch_size=optimal_batch_size, 
                        early_stopping_epochs=early_stopping_epochs,
                        train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                        step=step, configure_optimizers=configure_optimizers)

