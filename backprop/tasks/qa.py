from typing import List, Tuple, Union, Dict
from .base import Task
from backprop.models import T5QASummaryEmotion, BaseModel

import requests

DEFAULT_LOCAL_MODEL = T5QASummaryEmotion

LOCAL_MODELS = {
    "english": DEFAULT_LOCAL_MODEL
}

DEFAULT_API_MODEL = "english"

FINETUNABLE_MODELS = ["t5", "t5-base-qa-summary-emotion"]

API_MODELS = ["english"]

class QA(Task):
    """
    Task for Question Answering.

    Attributes:
        model:
            1. Name of the model on Backprop's qa endpoint (english or your own uploaded model)
            2. Officially supported local models (english).
            3. Model class of instance Backprop's BaseModel that implements the qa task
            4. Path/name of saved Backprop model
        model_class (optional): The model class to use when supplying a path for the model.
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
    
    def __call__(self, question: Union[str, List[str]], context: Union[str, List[str]],
                prev_qa: Union[List[Tuple[str, str]], List[List[Tuple[str, str]]]] = []):
        """Perform QA, either on docstore or on provided context.

        Args:
            question: Question (string or list of strings) for qa model.
            context: Context (string or list of strings) to ask question from.
            prev_qa (optional): List of previous question, answer tuples or list of prev_qa.

        Returns:
            Answer string or list of answer strings
        """

        prev_q = []
        prev_a = []
        if prev_qa != [] and type(prev_qa[0]) == list:
            for prev_qa in prev_qa:
                prev_q += [q for q, a in prev_qa]
                prev_a += [a for q, a in prev_qa]
        else:
            prev_q = [q for q, a in prev_qa]
            prev_a = [a for q, a in prev_qa]

        task_input = {
            "question": question,
            "context": context,
            "prev_q": prev_q,
            "prev_a": prev_a,
        }
        if self.local:
            return self.model(task_input, task="qa")
        else:
            task_input["model"] = self.model

            res = requests.post("https://api.backprop.co/qa", json=task_input,
                                headers={"x-api-key": self.api_key}).json()

            if res.get("message"):
                raise Exception(f"Failed to make API request: {res['message']}")

            return res["answer"]
    
    def finetune(self, params: Dict, *args, **kwargs):
        """
        Passes args and kwargs to the model's finetune method.
        Input orderings must match.

        Args:
            params: dictionary of lists: 'questions', 'answers', 'contexts'.
                    Optionally includes 'prev_qas': list of list of (q, a) tuples to prepend to context.
        
        Examples::

            import backprop
            
            # Initialise task
            qa = backprop.QA(backprop.models.T5)

            questions = ["What's Backprop?", "What language is it in?", "When was the Moog synthesizer invented?"]
            answers = ["A library that trains models", "Python", "1964"]
            contexts = ["Backprop is a Python library that makes training and using models easier.", 
                        "Backprop is a Python library that makes training and using models easier.",
                        "Bob Moog was a physicist. He invented the Moog synthesizer in 1964."]
            
            prev_qas = [[], 
                        [("What's Backprop?", "A library that trains models")],
                        []]

            params = {"questions": questions,
                      "answers": answers,
                      "contexts": contexts,
                      "prev_qas": prev_qas}

            # Finetune
            qa.finetune(params=params)
        """
        # params = kwargs["params"]
        print(params)
        if not "questions" in params:
            print("Params requires key: 'questions' (list of questions)")
            return
        if not "answers" in params:
            print("Params requires key: 'answers' (list of answers)")
            return
        if not "contexts" in params:
            print("Params requires key: 'contexts' (list of question contexts)")
            return

        try:
            return self.model.finetune(params=params, task="qa", *args, **kwargs)
        except NotImplementedError:
            raise NotImplementedError(f"This model does not support finetuning, try: {', '.join(FINETUNABLE_MODELS)}")