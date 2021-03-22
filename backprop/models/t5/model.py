from typing import List, Tuple
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers.optimization import Adafactor
import torch
from torch.utils.data import DataLoader
from random import shuffle
import os

from backprop.models import TextGenerationModel, Finetunable

class T5(TextGenerationModel, Finetunable):
    """
    Google's T5 model for text-generation.

    Attributes:
        args: args passed to :class:`backprop.models.generic_models.TextGenerationModel`
        model_path: path to a T5 model on huggingface (t5-small, t5-base, t5-large)
        kwargs: kwargs passed to :class:`backprop.models.generic_models.TextGenerationModel`
    """
    def __init__(self, *args, model_path="t5-small", **kwargs):
        Finetunable.__init__(self)
        TextGenerationModel.__init__(self, model_path,
                                *args, **kwargs)

        self.batch_size = 1
        self.tasks = ["text-generation", "generation"]
        self.description = "This is the T5 model by Google."
        self.name = "t5"

    def __call__(self, task_input, task="text-generation"):
        """
        Uses the model for the text-generation task

        Args:
            task_input: input dictionary according to the ``text-generation`` task specification,
                        or specification of task you've finetuned for.
            task: text-generation, or a task you've tuned a T5 model on
        """
        if task in self.tasks:
            if task in ["text-generation", "generation"]:
                text = task_input.pop("text")
                return self.generate(text, **task_input)
            elif task == "emotion":
                return self.emotion(task_input["text"])
            elif task == "summarisation":
                return self.summarise(task_input["text"])
            elif task == "qa":
                prev_q = task_input.get("prev_q", [])
                prev_a = task_input.get("prev_a", [])
                prev_qa = []
                if len(prev_q) != 0:
                    prev_qa = list(zip(prev_q, prev_a))
                return self.qa(task_input["question"], task_input["context"], prev_qa=prev_qa)

        else:
            raise ValueError(f"Unsupported task: {task}")

    def process_qa(self, question, context, prev_qa):
        input_text = [f"q: {qa[0]} a: {qa[1]}" for qa in prev_qa]
        input_text.append(f"q: {question}")
        input_text.append(f"c: {context}")
        input_text = " ".join(input_text)

        return input_text


    def qa(self, question, context, prev_qa: List[Tuple[str, str]] = []):
        if isinstance(question, list):
            # Must have a consistent amount of examples
            assert(len(question) == len(context))
            if len(prev_qa) != 0:
                assert(len(question) == len(prev_qa))
            else:
                prev_qa = [prev_qa] * len(question)

            # Process according to the model used
            input_text = [self.process_qa(q, c, p)
                          for q, c, p in zip(question, context, prev_qa)]
        else:
            input_text = self.process_qa(question, context, prev_qa)

        return self.generate(input_text, do_sample=False, max_length=96)

    
    def process_emotion(self, text):
        return f"emotion: {text}"

    
    def emotion(self, text):
        if isinstance(text, list):
            # Process according to the model used
            text = [self.process_emotion(item) for item in text]
        else:
            text = self.process_emotion(text)

        return self.generate(text, do_sample=False, max_length=96)

    
    def process_summarisation(self, text):
        return f"summarise: {text}"

    
    def summarise(self, text):
        if isinstance(text, list):
            # Process according to the model used
            text = [self.process_summarisation(item) for item in text]
        else:
            text = self.process_summarisation(text)

        return self.generate(text, do_sample=False, max_length=96)
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return Adafactor(params=self.model.parameters(), lr=1e-3, scale_parameter=False, relative_step=False)

    def encode(self, row, max_input_length, max_output_length):
        inp = self.encode_input(row[0])
        out = self.encode_output(row[1])

        row = {**inp, **out}
        return row
    
    def encode_input(self, text, max_length=128):
        tokens = self.tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        return {"input_ids": tokens.input_ids[0], "attention_mask": tokens.attention_mask[0]}

    def encode_output(self, text, max_length=32):
        tokens =  self.tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        return {"labels": tokens.input_ids[0], "decoder_attention_mask": tokens.attention_mask[0]}
    
    def prepare_qa(self, questions, answers, contexts, prev_qas):
        assert len(questions) == len(answers) and len(answers) == len(contexts)
        if prev_qas:
            assert len(prev_qas) == len(questions)
        
        questions = [f"q: {q}" for q in questions]
        contexts = [f"context: {c}" for c in contexts]
        answers = [f"{a}" for a in answers]
        
        # Creates previous QA strings 
        prev_qas_prep = []
        if prev_qas:
            for prev_qa in prev_qas:
                if prev_qa != []:
                     qa_str = [f"q: {q}\na: {a}" for q,a in prev_qa]
                     qa_str = "".join(qa_str)
                     prev_qas_prep.append(qa_str)
                else:
                    prev_qas_prep.append("")

        inps = []
        for i in range(len(questions)):
            prev = "" if not prev_qas else prev_qas_prep[i]
            inp = f"{prev}\n{questions[i]}\n{contexts[i]}\na: "
            inps.append(inp)
        
        return inps, answers

    def finetune(self, params, max_input_length=128, max_output_length=32,
                 validation_split: float=0.15, epochs: int=20,
                 batch_size: int=None, early_stopping: bool = True,
                 trainer: pl.Trainer = None, task: str = "text-generation"):
        """
        Finetunes T5 for the text-generation task.
        
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

        if task == "text-generation":
            pass
        elif task == "summarisation":
            params["input_text"] = [f"summarise: {i}" for i in params["input_text"]]
            self.tasks.append(task)
        elif task == "emotion":
            params["input_text"] = [f"emotion: {i}" for i in params["input_text"]]
            self.tasks.append(task)
        elif task == "qa":
            prev_qas = None
            if "prev_qas" in params:
                prev_qas = params["prev_qas"]
            inps, outs = self.prepare_qa(params["questions"], params["answers"], params["contexts"], prev_qas)
            params["input_text"] = inps
            params["output_text"] = outs
            self.tasks.append(task)
        else:
            raise ValueError(f"Unsupported task: {task}")

        assert len(params["input_text"]) == len(params["output_text"])
        OPTIMAL_BATCH_SIZE = 128                
        
        print("Processing data...")
        dataset = zip(params["input_text"], params["output_text"])
        dataset = [self.encode(r, max_input_length, max_output_length) for r in dataset]

        Finetunable.finetune(self, dataset, validation_split=validation_split,
            epochs=epochs, batch_size=batch_size, optimal_batch_size=OPTIMAL_BATCH_SIZE,
            early_stopping=early_stopping, trainer=trainer)