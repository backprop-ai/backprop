from typing import List, Tuple, Union, Dict
from .base import Task
from backprop.models import BaseModel, AutoModel
from transformers.optimization import Adafactor
from backprop.utils.datasets import TextToTextDataset

import requests

TASK = "summarisation"

DEFAULT_LOCAL_MODEL = "t5-base-qa-summary-emotion"

LOCAL_ALIASES = {
    "english": "t5-base-qa-summary-emotion"
}

class Summarisation(Task):
    """
    Task for summarisation.

    Attributes:
        model:
            1. Model name
            2. Model name on Backprop's summarisation endpoint
            3. Model object that implements the summarisation task
        local (optional): Run locally. Defaults to False
        api_key (optional): Backprop API key for non-local inference
        device (optional): Device to run inference on. Defaults to "cuda" if available.
    """
    def __init__(self, model: Union[str, BaseModel] = None,
                local: bool = False, api_key: str = None, device: str = None):
        models = AutoModel.list_models(task=TASK)

        super().__init__(model, local=local, api_key=api_key, device=device,
                        models=models, task=TASK,
                        default_local_model=DEFAULT_LOCAL_MODEL)
    
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

    def __call__(self, text: Union[str, List[str]]):
        """Perform summarisation on input text.

        Args:
            text: string or list of strings to be summarised - keep each string below 500 words.

        Returns:
            Summary string or list of summary strings.
        """
        task_input = {
            "text": text
        }
        if self.local:
            return self.model(task_input, task="summarisation")
        else:
            task_input["model"] = self.model

            res = requests.post("https://api.backprop.co/summarisation", json=task_input,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            return res["summary"]
    
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
        Returns default optimizer for summarisation (AdaFactor, learning rate 1e-3)
        """
        return Adafactor(params=self.model.parameters(), lr=1e-3, scale_parameter=False, relative_step=False)
    
    def finetune(self, params, validation_split: Union[float, Tuple[List[int], List[int]]]=0.15,
                  max_input_length: int=512, max_output_length: int=128,
                  epochs: int=20, batch_size: int=None,
                  optimal_batch_size: int=None, early_stopping_epochs: int=1,
                  train_dataloader=None, val_dataloader=None, step=None,
                  configure_optimizers=None):
        """
        Finetunes a generative model for summarisation.
        
        Note:
            input_text and output_text in params must have matching ordering (item 1 of input must match item 1 of output)

        Args:
            params: Dictionary of model inputs. Contains 'input_text' and 'output_text' keys, with values as lists of input/output data.
            max_input_length: Maximum number of tokens (1 token ~ 1 word) in input. Anything higher will be truncated. Max 512.
            max_output_length: Maximum number of tokens (1 token ~ 1 word) in output. Anything higher will be truncated. Max 512.
            validation_split: Float between 0 and 1 that determines what percentage of the data to use for validation
            epochs: Integer specifying how many training iterations to run
            batch_size: Batch size when training. Leave as None to automatically determine batch size.
            optimal_batch_size: Optimal batch size for the model being trained -- defaults to model settings.
            early_stopping_epochs: Integer determining how many epochs will run before stopping without an improvement in validation loss
            train_dataloader: Dataloader for providing training data when finetuning. Defaults to inbuilt dataloder.
            val_dataloader: Dataloader for providing validation data when finetuning. Defaults to inbuilt dataloader.
            step: Function determining how to call model for a training step. Defaults to step defined in this task class.
            configure_optimizers: Function that sets up the optimizer for training. Defaults to optimizer defined in this task class.
        
        Examples::
            
            import backprop

            summary = backprop.Summarisation()

            # Provide training data for task
            inp = ["This is a long news article about recent political happenings.", "This is an article about some recent scientific research."]
            out = ["Short political summary.", "Short scientific summary."]
            params = {"input_text": inp, "output_text": out}

            # Finetune
            summary.finetune(params)
        """
        inputs = params["input_text"]
        outputs = params["output_text"]
        assert len(inputs) == len(outputs)

        step = step or self.step
        configure_optimizers = configure_optimizers or self.configure_optimizers
        
        dataset_params = {
            "input": inputs,
            "output": outputs,
            "max_input_length": max_input_length,
            "max_output_length": max_output_length
        }

        print("Processing data...")
        dataset = TextToTextDataset(dataset_params, task=TASK, process_batch=self.model.process_batch, length=len(inputs))

        super().finetune(dataset=dataset, validation_split=validation_split,
                        epochs=epochs, batch_size=batch_size, optimal_batch_size=optimal_batch_size,
                        early_stopping_epochs=early_stopping_epochs,
                        train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                        step=step, configure_optimizers=configure_optimizers)
        



