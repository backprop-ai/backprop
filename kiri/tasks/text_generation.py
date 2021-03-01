from typing import List, Tuple, Union
from kiri.models import GPT2Large, T5QASummaryEmotion, BaseModel
from .base import Task

import requests

DEFAULT_LOCAL_MODEL = GPT2Large

LOCAL_MODELS = {
    "gpt2": DEFAULT_LOCAL_MODEL,
    "t5-base-qa-summary-emotion": T5QASummaryEmotion
}

DEFAULT_API_MODEL = "gpt2-large"

API_MODELS = ["gpt2-large", "t5-base-qa-summary-emotion"]

class TextClassification(Task):
    """
    Task for text generation.

    Attributes:
        model:
            1. Name of the model on Kiri's generation endpoint (gpt2-large, t5-base-qa-summary-emotion)
            2. Officially supported local models (gpt2, t5-base-qa-summary-emotion) or Huggingface path to the model.
            3. Kiri's GenerationModel object
        local (optional): Run locally. Defaults to True
        api_key (optional): Kiri API key for non-local inference
        device (optional): Device to run inference on. Defaults to "cuda" if available.
        init (optional): Whether to initialise model immediately or wait until first call.
            Defaults to False
    """
    def __init__(self, model: Union[str, BaseModel] = None,
                local: bool = False, api_key: str = None, device: str = "cpu",
                init: bool = False):

        super().__init__(model, local=local, api_key=api_key, device=device,
                        init=init, local_models=LOCAL_MODELS, api_models=API_MODELS,
                        default_local_model=DEFAULT_LOCAL_MODEL,
                        default_api_model=DEFAULT_API_MODEL)

    
    def __call__(self, text, min_length=10, max_length=20, temperature=1.0,
                top_k=0.0, top_p=1.0, repetition_penalty=1.0, length_penalty=1.0,
                num_beams=1, num_generations=1, do_sample=True):
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
        if self.local:
            # task_input = {
            #     "text": text,
            # }
            # TODO: User proper task interface
            return self.model.generate(text,
                        min_length=min_length,
                        max_length=max_length, temperature=temperature,
                        top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty, num_beams=num_beams,
                        num_return_sequences=num_generations, do_sample=do_sample)
        else:
            body = {
                "text": text,
                "model": self.model,
                "min_length": min_length,
            }

            res = requests.post("https://api.kiri.ai/generation", json=body,
                                headers={"x-api-key": self.api_key}).json()

            return res["output"]