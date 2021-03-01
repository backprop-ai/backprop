from transformers import AutoModelForSequenceClassification, AutoTokenizer
from kiri.models import TextGenerationModel

class T5QASummaryEmotion(TextGenerationModel):
    def __init__(self, *args, **kwargs):
        return super().__init__("kiri-ai/t5-base-qa-summary-emotion",
                                *args, **kwargs)

        self.tasks = ["text-generation", "emotion", "summarisation", "qa"]
        self.description = "This is the T5 base model by Google, and has been finetuned further for Q&A, Summarisation, and Sentiment analysis (emotion detection)."
        self.name = "t5-base-qa-summary-emotion"

    def __call__(self, task_input, task="text-generation"):
        if task == "text-generation":
            text = task_input.pop("text")
            do_sample = True

            min_length = task_input.get("min_length") or 10
            max_length = task_input.get("max_length") or 20
            temperature = task_input.get("temperature") or 0.0
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