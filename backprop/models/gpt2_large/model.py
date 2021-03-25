from transformers import AutoModelForSequenceClassification, AutoTokenizer
from backprop.models import TextGenerationModel

class GPT2Large(TextGenerationModel):
    def __init__(self, *args, **kwargs):
        TextGenerationModel.__init__(self, "gpt2-large",
                                *args, **kwargs)

        self.tasks = ["text-generation"]
        self.description = "A large (774M parameter) version of OpenAI's GPT-2 model. This is a general-use model, and has not been further finetuned on any specific languages or tasks."
        self.name = "gpt2-large"

    def __call__(self, task_input, task="text-generation"):
        """
        Uses the model for the text-generation task

        Args:
            task_input: input dictionary according to the ``text-generation`` task specification.
            task: text-generation
        """
        if task in ["text-generation", "generation"]:
            text = task_input.pop("text")
            temperature = task_input.pop("temperature", 0.7)

            return self.generate(text, **task_input, pad_token_id=50256, temperature=temperature)
        else:
            raise ValueError(f"Unsupported task: {task}")