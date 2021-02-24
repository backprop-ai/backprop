from typing import List, Tuple, Union
from .models import GenerationModel, PathModel
from .clip import clip, simple_tokenizer
from PIL import Image
import torch

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


class CLIP(PathModel):
    def __init__(self, model_path="ViT-B/32", init_model=clip.load,
                init_tokenizer=simple_tokenizer.SimpleTokenizer, device=None, init=True):
        self.initialised = False
        self.init_model = init_model
        self.init_tokenizer = init_tokenizer
        self.model_path = model_path
        self.device = device

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialise
        if init:
            self.model, self.transform = self.init_model(model_path, device=self.device)
            self.tokenizer = self.init_tokenizer()
            
            self.initialised = True

    def __call__(self, image_path: str, labels: List[str]):
        return self.image_classification(image_path=image_path, labels=labels)        

    @torch.no_grad()
    def image_classification(self, image_path: Union[str, List[str]], labels: Union[List[str], List[List[str]]]):
        # TODO: Rename image_path to image, as it accepts BytesIO as well
        self.check_init()
        # TODO: Implement batching
        image = self.transform(Image.open(image_path)).unsqueeze(0).to(self.device)
        text = clip.tokenize(self.tokenizer, labels).to(self.device)

        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)
        
        logits_per_image, logits_per_text = self.model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy().tolist()[0]

        label_probs = zip(labels, probs)
        probabilities = {lp[0]: lp[1] for lp in label_probs}
        return probabilities