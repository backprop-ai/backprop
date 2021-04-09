from typing import List, Tuple, Union, Dict
from .base import Task
from backprop.models import BaseModel, AutoModel
from transformers.optimization import Adafactor
from backprop.utils.datasets import TextToTextDataset

import requests

DEFAULT_LOCAL_MODEL = "t5-base-qa-summary-emotion"


class QA(Task):
    """
    Task for Question Answering.

    Attributes:
        model:
            1. Model name
            2. Model name on Backprop's qa endpoint
            3. Model object that implements the qa task
        local (optional): Run locally. Defaults to False
        api_key (optional): Backprop API key for non-local inference
        device (optional): Device to run inference on. Defaults to "cuda" if available.
    """
    def __init__(self, model: Union[str, BaseModel] = None,
                local: bool = False, api_key: str = None, device: str = None):
        task = "qa"
        models = AutoModel.list_models(task=task)

        super().__init__(model, local=local, api_key=api_key, device=device,
                        models=models, task=task,
                        default_local_model=DEFAULT_LOCAL_MODEL)
    
    @staticmethod
    def list_models(return_dict=False, display=False, limit=None):
        return AutoModel.list_models(task="qa", return_dict=return_dict, display=display, limit=limit)

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
            for pqa in prev_qa:
                if len(pqa) == 0:
                    prev_q.append([])
                    prev_a.append([])
                else:
                    q = []
                    a = []
                    for x in pqa:
                        q.append(x[0])
                        a.append(x[1])
                    prev_q.append(q)
                    prev_a.append(a)
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
    
    # def finetune(self, params: Dict, *args, **kwargs):
    #     """
    #     Passes args and kwargs to the model's finetune method.
    #     Input orderings must match.

    #     Args:
    #         params: dictionary of lists: 'questions', 'answers', 'contexts'.
    #                 Optionally includes 'prev_qas': list of list of (q, a) tuples to prepend to context.
        
    #     Examples::

    #         import backprop
            
    #         # Initialise task
    #         qa = backprop.QA(backprop.models.T5)

    #         questions = ["What's Backprop?", "What language is it in?", "When was the Moog synthesizer invented?"]
    #         answers = ["A library that trains models", "Python", "1964"]
    #         contexts = ["Backprop is a Python library that makes training and using models easier.", 
    #                     "Backprop is a Python library that makes training and using models easier.",
    #                     "Bob Moog was a physicist. He invented the Moog synthesizer in 1964."]
            
    #         prev_qas = [[], 
    #                     [("What's Backprop?", "A library that trains models")],
    #                     []]

    #         params = {"questions": questions,
    #                   "answers": answers,
    #                   "contexts": contexts,
    #                   "prev_qas": prev_qas}

    #         # Finetune
    #         qa.finetune(params=params)
    #     """
    #     # params = kwargs["params"]
    #     print(params)
    #     if not "questions" in params:
    #         print("Params requires key: 'questions' (list of questions)")
    #         return
    #     if not "answers" in params:
    #         print("Params requires key: 'answers' (list of answers)")
    #         return
    #     if not "contexts" in params:
    #         print("Params requires key: 'contexts' (list of question contexts)")
    #         return

    #     try:
    #         return self.model.finetune(params=params, task="qa", *args, **kwargs)
    #     except NotImplementedError:
    #         raise NotImplementedError(f"This model does not support finetuning, try: {', '.join(FINETUNABLE_MODELS)}")

    def step(self, batch, batch_idx):
        return self.model.training_step(batch)
        
    def configure_optimizers(self):
        return Adafactor(params=self.model.parameters(), lr=1e-3, scale_parameter=False, relative_step=False)

    def finetune(self, params, validation_split: Union[float, Tuple[List[int], List[int]]]=0.15,
                  max_input_length: int=256, max_output_length: int=32,
                  epochs: int=20, batch_size: int=None,
                  optimal_batch_size: int=None, early_stopping_epochs: int=1,
                  train_dataloader=None, val_dataloader=None, step=None,
                  configure_optimizers=None):
        
        questions = params["questions"]
        contexts = params["contexts"]
        answers = params["answers"]
        prev_qas = params["prev_qas"]
        
        assert len(questions) == len(answers) and len(questions) == len(contexts)
    
        step = step or self.step
        configure_optimizers = configure_optimizers or self.configure_optimizers

        dataset_params = {
            "question": questions,
            "context": contexts,
            "prev_qa": prev_qas,
            "output": answers,
            "max_input_length": max_input_length,
            "max_output_length": max_output_length
        }

        print("Processing data...")
        # dataset = QADataset(questions, contexts, prev_qas, answers, self.model.process_qa, max_input_length, max_output_length)
        dataset = TextToTextDataset(dataset_params, task="qa", process_batch=self.model.process_batch, length=len(questions))
        
        super().finetune(dataset=dataset, validation_split=validation_split,
                epochs=epochs, batch_size=batch_size, optimal_batch_size=optimal_batch_size,
                early_stopping_epochs=early_stopping_epochs,
                train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                step=step, configure_optimizers=configure_optimizers)
        