from typing import List, Tuple, Union, Dict
from backprop.models import GPT2Large, T5QASummaryEmotion, BaseModel, T5
from .base import Task
from backprop.utils.datasets import TextToTextDataset

import requests
from transformers.optimization import Adafactor

DEFAULT_LOCAL_MODEL = GPT2Large

LOCAL_MODELS = {
    "gpt2": DEFAULT_LOCAL_MODEL,
    "t5-base-qa-summary-emotion": T5QASummaryEmotion,
    "t5": T5
}

DEFAULT_API_MODEL = "gpt2-large"

FINETUNABLE_MODELS = ["t5", "t5-base-qa-summary-emotion"]

API_MODELS = ["gpt2-large", "t5-base-qa-summary-emotion"]

class TextGeneration(Task):
    """
    Task for text generation.

    Attributes:
        model:
            1. Name of the model on Backprop's generation endpoint (gpt2-large, t5-base-qa-summary-emotion or your own uploaded model)
            2. Officially supported local models (gpt2, t5-base-qa-summary-emotion).
            3. Model object/class that inherits from backprop's TextGenerationModel
            4. Path/name of saved Backprop model
        local (optional): Run locally. Defaults to False
        api_key (optional): Backprop API key for non-local inference
        device (optional): Device to run inference on. Defaults to "cuda" if available.
    """
    def __init__(self, model: Union[str, BaseModel] = None,
                local: bool = False, api_key: str = None, device: str = None):

        super().__init__(model, local=local, api_key=api_key, device=device,
                        local_models=LOCAL_MODELS, api_models=API_MODELS,
                        default_local_model=DEFAULT_LOCAL_MODEL,
                        default_api_model=DEFAULT_API_MODEL)

    
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
            return self.model(task_input, task="text-generation")
        else:
            task_input["model"] = self.model 

            res = requests.post("https://api.backprop.co/text-generation", json=task_input,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            return res["output"]

    def step(self, batch, batch_idx):
        return self.model.training_step(batch)

    def configure_optimizers(self):
        return Adafactor(params=self.model.parameters(), lr=1e-3, scale_parameter=False, relative_step=False)

    def finetune(self, params, validation_split: Union[float, Tuple[List[int], List[int]]] = 0.15,
                max_input_length: int = 128, max_output_length: int = 32,
                epochs: int = 20, batch_size: int = None,
                optimal_batch_size: int = None, early_stopping_epochs: int = 1,
                train_dataloader = None, val_dataloader = None, step = None,
                configure_optimizers = None):
        """
        TODO: Update
        Finetunes the model's text-generation task.
        
        Note:
            input_text and output_text in params must have matching ordering (item 1 of input must match item 1 of output)

        Args:
            params: Dictionary of model inputs. Contains 'input_text' and 'output_text' for generation, summarisation, and emotion.
                    QA requires specifically formatted input: see QA task's finetuning function for more details.
            max_input_length: Maximum number of tokens (1 token ~ 1 word) in input. Anything higher will be truncated. Max 512.
            max_output_length: Maximum number of tokens (1 token ~ 1 word) in output. Anything higher will be truncated. Max 512.
            validation_split: Float between 0 and 1 that determines what percentage of the data to use for validation
            epochs: Integer that specifies how many iterations of training to do
            batch_size: Leave as None to determine the batch size automatically
            early_stopping: Boolean that determines whether to automatically stop when validation loss stops improving
            trainer: Your custom pytorch_lightning trainer
            task: Task on which finetuning will occur. Must be in ["text-generation", "summarisation", "emotion", "qa"]

        Examples::

            import backprop
            
            # Initialise model
            model = backprop.models.T5()

            # Any text works as training data
            inp = ["I really liked the service I received!", "Meh, it was not impressive."]
            out = ["positive", "negative"]
            params = {"input_text": inp, "output_text": out}

            # Finetune
            model.finetune(params)
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
        dataset = TextToTextDataset(dataset_params, task="text-generation", process_batch=self.model.process_batch, length=len(input))

        super().finetune(dataset=dataset, validation_split=validation_split, epochs=epochs,
                batch_size=batch_size, optimal_batch_size=optimal_batch_size,
                early_stopping_epochs=early_stopping_epochs, step=step,
                configure_optimizers=configure_optimizers, train_dataloader=train_dataloader,
                val_dataloader=val_dataloader)