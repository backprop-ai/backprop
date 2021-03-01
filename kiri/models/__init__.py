from .tasks.qa import QA
from .tasks.summarisation import Summarisation
from .tasks.emotion import Emotion
from .tasks.classification import Classification
from .tasks.image_classification import ImageClassification
from .tasks.vectorisation import Vectorisation
from .tasks.generation import Generation
from .custom_models import T5QASummaryEmotion, CLIP
from .models import BaseModel, PathModel, HuggingModel, TextGenerationModel, ClassificationModel, VectorisationModel
from .bart_large_mnli import BartLargeMNLI
from .xlmr_large_xnli import XLMRLargeXNLI
from .gpt2_large import GPT2Large
from .t5_base_qa_summary_emotion import T5QASummaryEmotion