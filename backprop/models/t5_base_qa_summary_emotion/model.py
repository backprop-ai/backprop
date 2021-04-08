from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple
import torch

from backprop.models import T5

class T5QASummaryEmotion(T5):
    """
    Initialises a T5 model that has been finetuned on qa, summarisation and emotion detection.

    Attributes:
        args: args passed to :class:`backprop.models.t5.model.T5`
        model_path: path to an appropriate T5 model on huggingface (kiri-ai/t5-base-qa-summary-emotion)
        kwargs: kwrags passed to :class:`backprop.models.t5.model.T5`
    """
    def __init__(self, *args, model_path="kiri-ai/t5-base-qa-summary-emotion", **kwargs):
        T5.__init__(self, model_path=model_path,
                    *args, **kwargs)

        self.tasks = ["text-generation", "emotion", "summarisation", "qa"]
        self.description = "This is the T5 base model by Google, and has been finetuned further for Q&A, Summarisation, and Sentiment analysis (emotion detection)."
        self.name = "t5-base-qa-summary-emotion"

    def __call__(self, task_input, task="text-generation", train=False):
        """
        Uses the model for the chosen task

        Args:
            task_input: input dictionary according to the chosen task's specification
            task: one of text-generation, emotion, summarisation, qa 
        """
        if train:
            return self.model(**task_input)
        else:
            with torch.set_grad_enabled(train):
                if task in ["text-generation", "generation"]:
                    text = task_input.pop("text")
                    return self.generate(text, **task_input)
                elif task == "emotion":
                    return self.emote_or_summary(task_input["text"], "emotion")
                elif task == "summarisation":
                    return self.emote_or_summary(task_input["text"], "summarise")
                elif task == "qa":
                    prev_q = task_input.get("prev_q", [])
                    prev_a = task_input.get("prev_a", [])
                    prev_qa = []

                    if len(prev_q) != 0:
                        if type(prev_q[0]) != list:
                            qas = []
                            for x in range(len(prev_q)):
                                qas.append((prev_q[x], prev_a[x]))
                            prev_qa.append(qas)
                        else:
                            for pqa in zip(prev_q, prev_a):
                                if len(pqa[0]) == 0:
                                    prev_qa.append([])
                                else:
                                    qas = []
                                    for x in range(len(pqa[0])):
                                        pair = (pqa[0][x], pqa[1][x])
                                        qas.append(pair)
                                    prev_qa.append(qas)

                    return self.qa(task_input["question"], task_input["context"], prev_qa=prev_qa)
                else:
                    raise ValueError(f"Unsupported task: {task}")

    def emote_or_summary(self, text, task_prefix):
        if isinstance(text, list):
            text = [f"{task_prefix}: {t}" for t in text]
        else:
            text = f"{task_prefix}: {text}"
        
        return self.generate(text, do_sample=False, max_length=96)

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
        print(input_text)
        return self.generate(input_text, do_sample=False, max_length=96)
        
    def process_summarisation(self, inp, out, max_input_length, max_output_length):
        inp = f"summarise: {inp}"
        
        inp = self.encode_input(inp, max_input_length)
        out = self.encode_output(out, max_output_length)

        return inp, out

    def process_qa(self, question, context, prev_qa, answer=None, max_input_length=256, max_output_length=32):
        input_text = [f"q: {qa[0]} a: {qa[1]}" for qa in prev_qa]
        input_text.append(f"q: {question}")
        input_text.append(f"c: {context}")
        input_text = " ".join(input_text)
        
        if answer:
            input_text = self.encode_input(input_text, max_input_length)
            answer = self.encode_output(answer, max_output_length)
            return input_text, answer
        else:
            return input_text

    def encode_input(self, inp, max_length):
        tokens = self.tokenizer(inp, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        return {"input_ids": tokens.input_ids[0], "attention_mask": tokens.attention_mask[0]}
    
    def encode_output(self, out, max_length):
        tokens = self.tokenizer(out, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        return {"labels": tokens.input_ids[0], "decoder_attention_mask": tokens.attention_mask[0]}

    
    def process_emotion(self, inp, out, max_input_length, max_output_length):
        inp = f"emotion: {inp}"

        inp = self.encode_input(inp, max_input_length)
        out = self.encode_output(out, max_output_length)

        return inp, out