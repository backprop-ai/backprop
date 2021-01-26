from typing import List, Tuple
from .models import GenerationModel

class T5QASummaryEmotion(GenerationModel):
    """Custom class for Kiri's T5 model for QA, Summarisation, Emotion tasks.

    \*args and \*\*kwargs according to GenerationModel class.
    
    model_path defaults to "kiri-ai/t5-base-qa-summary-emotion"
    """
    def __init__(self, *args, **kwargs):
        return super().__init__("kiri-ai/t5-base-qa-summary-emotion",
                                *args, **kwargs)


    def process_qa(self, question, context, prev_qa):
        input_text = [f"q: {qa[0]} a: {qa[1]}" for qa in prev_qa]
        input_text.append(f"q: {question}")
        input_text.append(f"c: {context}")
        input_text = " ".join(input_text)

        return input_text


    def qa(self, question, context, prev_qa: List[Tuple[str, str]] = []):
        self.check_init()
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
        self.check_init()
        if isinstance(text, list):
            # Process according to the model used
            text = [self.process_emotion(item) for item in text]
        else:
            text = self.process_emotion(text)

        return self.generate(text, do_sample=False, max_length=96)

    
    def process_summarisation(self, text):
        return f"summarise: {text}"

    
    def summarise(self, text):
        self.check_init()
        if isinstance(text, list):
            # Process according to the model used
            text = [self.process_summarisation(item) for item in text]
        else:
            text = self.process_summarisation(text)

        return self.generate(text, do_sample=False, max_length=96)