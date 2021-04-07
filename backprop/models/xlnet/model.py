import torch
import pytorch_lightning as pl
from transformers import XLNetForSequenceClassification, XLNetTokenizer
from transformers.optimization import AdamW
from backprop.models import ClassificationModel, Finetunable
from typing import List, Union


class XLNet(ClassificationModel, Finetunable):
    """
    CMU & Google Brain's XLNet model for text classification.

    Attributes:
        args: args passed to :class:`backprop.models.generic_models.ClassificationModel`
        model_path: path to an XLNet model on hugging face (xlnet-base-cased, xlnet-large-cased)
        kwargs: kwargs passed to :class:`backprop.models.generic_models.ClassificationModel`

    """

    def __init__(self, *args, model_path="xlnet-base-cased", **kwargs):
        Finetunable.__init__(self)

        ClassificationModel.__init__(self, model_path, model_class=XLNetForSequenceClassification, 
                                    tokenizer_class=XLNetTokenizer, *args, **kwargs)

        self.tasks = ["text-classification"]
        self.description = "XLNet"
        self.name = "xlnet"
        self.pre_finetuning = self.init_pre_finetune

    def __call__(self, task_input, task="text-classification", train=False):
        """
        Uses the model for text classification.
        At this point, the model needs to already have been finetuned.
        This is what sets up the final layer for classification.

        Args:
            task_input: input dictionary according to the ``text-classification`` task specification
            task: text-classification
        """
    
        if task == "text-classification":
            if not train:
                with torch.set_grad_enabled(train):
                    text = task_input.pop("text")

                    is_list = type(text) == list
                    
                    text = text if is_list else [text]
                    
                    outputs = []
                    for t in text:
                        tokens = self.tokenizer(t, truncation=True, padding="max_length", return_tensors="pt")
                        input_ids = tokens.input_ids[0].unsqueeze(0).to(self._model_device)
                        mask = tokens.attention_mask[0].unsqueeze(0).to(self._model_device)
                        
                        output = self.model(input_ids=input_ids, attention_mask=mask)
                        outputs.append(output)
                    
                    outputs = outputs if is_list else outputs[0]

                    return outputs, self.labels
            else:
                return self.model(**task_input)

        else:
            raise ValueError(f"Unsupported task: {task}")
        

    def process_text(self, text, target, max_input_length):
        tokens = self.tokenizer(text, truncation=True, max_length=max_input_length, padding="max_length", return_tensors="pt")
        return {
            "input_ids": tokens.input_ids[0],
            "attention_mask": tokens.attention_mask[0],
            "labels": target
        }

    def encode(self, text, target, max_input_length=128):
        tokens = self.tokenizer(text, truncation=True, max_length=max_input_length, padding="max_length", return_tensors="pt")
        return tokens.input_ids[0], tokens.attention_mask[0], target

    def init_pre_finetune(self, labels):
        if not hasattr(self, "labels") or len(self.labels) != len(labels):
            self.labels = labels
            self.model = XLNetForSequenceClassification.from_pretrained(self.model_path, num_labels=len(labels))
