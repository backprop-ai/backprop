from typing import List, Tuple, Union, Dict
from backprop.models import AutoModel, BaseModel
from .base import Task
from backprop.utils.datasets import TextToTextDataset

import requests
from transformers.optimization import Adafactor

TASK = "text-generation"

DEFAULT_LOCAL_MODEL = "gpt2-medium"

LOCAL_ALIASES = {
    "english": "gpt2-medium",
    "gpt2": "gpt2-medium"
}

class TextGeneration(Task):
    """
    Task for text generation.

    Attributes:
        model:
            1. Model name
            2. Model name on Backprop's text-generation endpoint
            3. Model object that implements the text-generation task
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

    
    def __call__(self, text: Union[str, List[str]], min_length: int = None, max_length: int = None, temperature: float = None,
                top_k: int = None, top_p: float = None, repetition_penalty: float = None, length_penalty: float = None,
                num_beams: int = None, num_generations: int = None, do_sample: bool = None):
        """Generates text to continue from the given input.

        Args:
            input_text (string): Text from which the model will begin generating.
            min_length (int): Minimum number of tokens to generate (1 token ~ 1 word).
            max_length (int): Maximum number of tokens to generate (1 token ~ 1 word).
            temperature (float): Value that alters the randomness of generation (0.0 is no randomness, higher values introduce randomness. 0.5 - 0.7 is a good starting point).
            top_k (int): Only choose from the top_k tokens when generating (0 is no limit). 
            top_p (float): Only choose from the top tokens with combined probability greater than top_p.
            repetition_penalty (float): Penalty to be applied to tokens present in the input_text and
                tokens already generated in the sequence (>1 discourages repetition while <1 encourages).
            length_penalty (float): Penalty applied to overall sequence length. Set >1 for longer sequences,
                or <1 for shorter ones. 
            num_beams (int): Number of beams to be used in beam search. Does a number of generations to pick the best one. (1: no beam search)
            num_generations (int): How many times to run generation. Results are returned as a list. 
            do_sample (bool): Whether or not sampling strategies (temperature, top_k, top_p) should be used.

        Example::

            import backprop

            tg = backprop.TextGeneration()
            tg("Geralt knew the sings, the monster was a", min_length=20, max_length=50, temperature=0.7)
            > "Geralt knew the sings, the monster was a real danger, and he was the only one in the village who knew how to defend himself."
        """
        params = [("text", text), ("min_length", min_length), ("max_length", max_length),
                ("temperature", temperature), ("top_k", top_k), ("top_p", top_p),
                ("repetition_penalty", repetition_penalty), ("length_penalty", length_penalty),
                ("num_beams", num_beams), ("num_generations", num_generations),
                ("do_sample", do_sample)]
        
        # Ignore None to let the model decide optimal values
        task_input = {k: v for k, v in params if v != None}
        if self.local:
            return self.model(task_input, task=TASK)
        else:
            task_input["model"] = self.model 

            res = requests.post("https://api.backprop.co/text-generation", json=task_input,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            return res["output"]

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
        Returns default optimizer for text generation (AdaFactor, learning rate 1e-3)
        """
        return Adafactor(params=self.model.parameters(), lr=1e-3, scale_parameter=False, relative_step=False)

    def finetune(self, params, validation_split: Union[float, Tuple[List[int], List[int]]] = 0.15,
                max_input_length: int = 128, max_output_length: int = 32,
                epochs: int = 20, batch_size: int = None,
                optimal_batch_size: int = None, early_stopping_epochs: int = 1,
                train_dataloader = None, val_dataloader = None, step = None,
                configure_optimizers = None):
        """
        Finetunes a model for a text generation task.
        
        Note:
            input_text and output_text in params must have matching ordering (item 1 of input must match item 1 of output)

        Args:
            params: Dictionary of model inputs. Contains 'input_text' and 'output_text' keys, with values as lists of input/output data.
            max_input_length: Maximum number of tokens (1 token ~ 1 word) in input. Anything higher will be truncated. Max 512.
            max_output_length: Maximum number of tokens (1 token ~ 1 word) in output. Anything higher will be truncated. Max 512.
            validation_split: Float between 0 and 1 that determines what percentage of the data to use for validation.
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
            
            tg = backprop.TextGeneration()

            # Any text works as training data
            inp = ["I really liked the service I received!", "Meh, it was not impressive."]
            out = ["positive", "negative"]
            params = {"input_text": inp, "output_text": out}

            # Finetune
            tg.finetune(params)
        """
        input_text = params["input_text"]
        output_text = params["output_text"]
        assert len(input_text) == len(output_text), "The input lists must match"

        optimal_batch_size = getattr(self.model, "optimal_batch_size", 128)

        configure_optimizers = configure_optimizers or self.configure_optimizers

        step = step or self.step

        dataset_params = {
            "input": input_text,
            "output": output_text,
            "max_input_length": max_input_length,
            "max_output_length": max_output_length
        }

        print("Processing data...")
        dataset = TextToTextDataset(dataset_params, task=TASK, process_batch=self.model.process_batch, length=len(input_text))

        super().finetune(dataset=dataset, validation_split=validation_split, epochs=epochs,
                batch_size=batch_size, optimal_batch_size=optimal_batch_size,
                early_stopping_epochs=early_stopping_epochs, step=step,
                configure_optimizers=configure_optimizers, train_dataloader=train_dataloader,
                val_dataloader=val_dataloader)