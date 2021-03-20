import torch
import torch.nn.functional as F
from transformers import XLNetForSequenceClassification, XLNetTokenizer
from transformers.optimization import AdamW
from backprop.models import ClassificationModel, Finetunable
from typing import List, Union


class XLNet(ClassificationModel, Finetunable):
    def __init__(self, *args, model_path="xlnet-base-cased", **kwargs):
        Finetunable.__init__(self)

        ClassificationModel.__init__(self, model_path, model_class=XLNetForSequenceClassification, 
                                    tokenizer_class=XLNetTokenizer, *args, **kwargs)

        self.tasks = ["text-classification"]
        self.description = "XLNet"
        self.name = "xlnet"

    def __call__(self, task_input, task="text-classification"):
        if task == "text-classification":
            text = task_input.pop("text")
            
            return self.text_classification(text)
        else:
            raise ValueError(f"Unsupported task: {task}")
        
    @torch.no_grad()
    def text_classification(self, text: Union[str, List[str]]):
        self.model.to(self._device)

        is_list = type(text) == list
        
        if not is_list:
            text = [text]
        
        probabilities = []
        for t in text:
            tokens = self.tokenizer(t, truncation=True, padding="max_length", return_tensors="pt")
            
            input_ids = tokens.input_ids[0].unsqueeze(0).to(self._device)
            mask = tokens.attention_mask[0].unsqueeze(0).to(self._device)

            inp = {
                "input_ids": input_ids,
                "attention_mask": mask
            }

            outputs = self.model(**inp)
            logits = outputs[0]
            predictions = torch.softmax(logits, dim=1).detach().squeeze(0).tolist()
            probs = {}
            for idx, pred in enumerate(predictions):
                label = self.labels[idx]
                probs[label] = pred
            
            probabilities.append(probs)
        
        if not is_list:
            probabilities = probabilities[0]
        
        return probabilities

    def encode(self, text, target, max_input_length=128):
        tokens = self.tokenizer(text, truncation=True, max_length=max_input_length, padding="max_length", return_tensors="pt")
        return tokens.input_ids[0], tokens.attention_mask[0], target

    def training_step(self, batch, batch_idx):
        inputs, masks, targets = batch
        outputs = self.model(input_ids=inputs, attention_mask=masks, labels=targets)
        loss = outputs[0]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, masks, targets = batch
        outputs = self.model(input_ids=inputs, attention_mask=masks, labels=targets)
        loss = outputs[0]
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return AdamW(params=self.model.parameters(), lr=2e-5)
    
    def finetune(self, input_text: List[str], output: List[str],
                max_input_length=128, validation_split: float=0.15, 
                epochs: int=20):
        """
        Finetune XLNet for text classification.
        input_text and output must be ordered 1:1
        Unique data classes automatically determined from output data

        Args:
            input_text: List of strings to classify (must match output ordering)
            output: List of input classifications (must match input ordering)
            max_input_length: Length to cut off input text
            validation_split: Float between 0 and 1 that determines what percentage of the data to use for validation
            epochs: Integer that specifies how many iterations of training to do
            batch_size: Leave as None to determine the batch size automatically
        """

        assert len(input_text) == len(output)
        OPTIMAL_BATCH_SIZE = 128

        labels = set(output)
        self.labels = {k: v for k, v in enumerate(labels)}
        
        class_to_idx = {v: k for k, v in enumerate(labels)}

        self.model = XLNetForSequenceClassification.from_pretrained(self.model_path, num_labels=len(labels))
        self.model.to(self._device)

        print("Processing data...")
        dataset = zip(input_text, output)
        dataset = [(self.encode(r[0], class_to_idx[r[1]], max_input_length)) for r in dataset]
        Finetunable.finetune(self, dataset, validation_split, epochs, optimal_batch_size=OPTIMAL_BATCH_SIZE)
