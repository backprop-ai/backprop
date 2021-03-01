from transformers import AutoModelForSequenceClassification, AutoTokenizer
from kiri.models import TextGenerationModel

class GPT2Large(TextGenerationModel):
    def __init__(self, model_path="gpt2-large", tokenizer_path=None,
                model_class=AutoModelForSequenceClassification,
                tokenizer_class=AutoTokenizer, device=None, init=True):
        TextGenerationModel.__init__(self, model_path, tokenizer_path=tokenizer_path,
                    model_class=model_class, tokenizer_class=tokenizer_class,
                    device=device, init=init)

        self.tasks = ["text-generation"]
        self.description = "A large (774M parameter) version of OpenAI's GPT-2 model. This is a general-use model, and has not been further finetuned on any specific languages or tasks."
        self.name = "gpt2-large"

    def __call__(self, task_input, task="text-generation"):
        if task == "text-generation":
            text = task_input.pop("text")
            do_sample = True

            min_length = task_input.get("min_length") or 10
            max_length = task_input.get("max_length") or 20
            temperature = task_input.get("temperature") or 1.0
            top_k = task_input.get("top_k") or 0.0
            top_p = task_input.get("top_p") or 1.0
            repetition_penalty = task_input.get("repetition_penalty") or 1.0
            length_penalty = task_input.get("length_penalty") or 1.0
            num_beams = task_input.get("num_beams") or 1
            num_generations = task_input.get("num_generations") or 1

            return self.generate(text, min_length=min_length, max_length=max_length,
                            temperature=temperature, top_k=top_k, top_p=top_p,
                            repetition_penalty=repetition_penalty,
                            length_penalty=length_penalty, num_beams=num_beams,
                            num_return_sequences=num_generations, do_sample=do_sample)
        else:
            raise ValueError(f"Unsupported task: {task}")