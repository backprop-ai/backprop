from typing import List, Tuple, Union
from backprop.models import GPT2Large, T5QASummaryEmotion, BaseModel, T5
from .base import Task

import requests

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

    
    def __call__(self, text: Union[str, List[str]], min_length=None, max_length=None, temperature=None,
                top_k=None, top_p=None, repetition_penalty=None, length_penalty=None,
                num_beams=None, num_generations=None, do_sample=None):
        """Generates text to continue off the given input.

        Args:
            input_text: Text from which model will begin generating.
            min_length: Minimum length of generation before EOS can be generated.
            max_length: Maximum length of generated sequence.
            temperature: Value that alters softmax probabilities.
            top_k: Sampling strategy in which probabilities are redistributed among top k most-likely words.
            top_p: Sampling strategy in which probabilities are distributed among 
                set of words with combined probability greater than p.
            repetition_penalty: Penalty to be applied to words present in the input_text and
                words already generated in the sequence.
            length_penalty: Penalty applied to overall sequence length. Set >1 for longer sequences,
                or <1 for shorter ones. 
            num_beams: Number of beams to be used in beam search. (1: no beam search)
            num_generations: How many times to run generation. 
            do_sample: Whether or not sampling strategies (top_k & top_p) should be used.
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

    def finetune(self, *args, **kwargs):
        """
        Passes the args and kwargs to the model's finetune method.
        """
        try:
            return self.model.finetune(*args, **kwargs)
        except NotImplementedError:
            raise NotImplementedError(f"This model does not support finetuning, try: {', '.join(FINETUNABLE_MODELS)}")