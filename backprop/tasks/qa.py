from typing import List, Tuple, Union, Dict
from .base import Task
from backprop.models import BaseModel, AutoModel
from transformers.optimization import Adafactor
from backprop.utils.datasets import TextToTextDataset

import requests

TASK = "qa"

DEFAULT_LOCAL_MODEL = "t5-base-qa-summary-emotion"

LOCAL_ALIASES = {
    "english": "t5-base-qa-summary-emotion"
}


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
        models = AutoModel.list_models(task=TASK)

        super().__init__(model, local=local, api_key=api_key, device=device,
                        models=models, task=TASK,
                        default_local_model=DEFAULT_LOCAL_MODEL,
                        local_aliases=LOCAL_ALIASES)
    
    @staticmethod
    def list_models(return_dict=False, display=False, limit=None):
        return AutoModel.list_models(task=TASK, return_dict=return_dict, display=display, limit=limit, aliases=LOCAL_ALIASES)

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
        """
        Passes args and kwargs to the model's finetune method.
        Input orderings must match.

        Args:
            params: dictionary of lists: 'questions', 'answers', 'contexts'.
                    Optionally includes 'prev_qas': list of lists containing (q, a) tuples to prepend to context.
            max_input_length: Maximum number of tokens (1 token ~ 1 word) in input. Anything higher will be truncated. Max 512.
            max_output_length: Maximum number of tokens (1 token ~ 1 word) in output. Anything higher will be truncated. Max 512.
            validation_split: Float between 0 and 1 that determines what percentage of the data to use for validation.
            epochs: Integer specifying how many training iterations to run
            batch_size: Batch size when training. Leave as None to automatically determine batch size.
            optimal_batch_size: Optimal batch size for the model being trained -- defaults to model settings.
            early_stopping_epochs: Integer determining how many epochs will run before stopping without an improvement in validation loss.
            train_dataloader: Dataloader for providing training data when finetuning. Defaults to inbuilt dataloder.
            val_dataloader: Dataloader for providing validation data when finetuning. Defaults to inbuilt dataloader.
            step: Function determining how to call model for a training step. Defaults to step defined in this task class.
            configure_optimizers: Function that sets up the optimizer for training. Defaults to optimizer defined in this task class. 
        

        Examples::

            import backprop
            
            # Initialise task
            qa = backprop.QA()

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

        # dataset = QADataset(questions, contexts, prev_qas, answers, self.model.process_qa, max_input_length, max_output_length)
        dataset = TextToTextDataset(dataset_params, task=TASK, process_batch=self.model.process_batch, length=len(questions))
        
        super().finetune(dataset=dataset, validation_split=validation_split,
                epochs=epochs, batch_size=batch_size, optimal_batch_size=optimal_batch_size,
                early_stopping_epochs=early_stopping_epochs,
                train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                step=step, configure_optimizers=configure_optimizers)
        