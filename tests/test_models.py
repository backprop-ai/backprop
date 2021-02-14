# TODO: Rename test_models.py to test_tasks.py and name imports better
from kiri.models import Vectorisation, QA, Summarisation, Emotion, Classification, ImageClassification
import numpy as np
import torch
import os

qa_example = {
    "q1": "Where does Sally live?",
    "q2": "How long has Sally lived in London?",
    "q2_conv": "How long has she lived there?",
    "q3": "Where did Sally live prior to London?",
    "q3_conv": "Where did she live before?",
    "c": "Sally has been living in London for 3 years. Previously, Sally lived in Liverpool.",
    "a1": "London",
    "a2": "3 years",
    "a3": "Liverpool"
}

summary_context = """Elon Reeve Musk FRS (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a business magnate, industrial designer and engineer.[6] He is the founder, CEO, CTO and chief designer of SpaceX; early investor,[b] CEO and product architect of Tesla, Inc.; founder of The Boring Company; co-founder of Neuralink; and co-founder and initial co-chairman of OpenAI. He was elected a Fellow of the Royal Society (FRS) in 2018.[9][10] Also that year, he was ranked 25th on the Forbes list of The World's Most Powerful People,[11] and was ranked joint-first on the Forbes list of the Most Innovative Leaders of 2019.[12] As of December 19, 2020, Musk’s net worth was estimated by Forbes to US$153.5 billion,[1][13] making him the second-richest person in the world, behind Jeff Bezos.[14]"""
classify_context = """I am mad because my product broke the first time I used it"""
classify_labels = ["product issue", "nature"]
classify_correct = "product issue"

main_path = os.path.dirname(__file__)
image_classification_image = os.path.join(main_path, "data/dog.png")
image_classification_labels = ["cat", "dog"]
image_classification_correct = "dog"

device = "cuda" if torch.cuda.is_available() else "cpu"

vectorise = Vectorisation(local=True, device=device)
qa = QA(local=True, device=device)
emotion = Emotion(local=True, device=device)
summarise = Summarisation(local=True, device=device)
classify = Classification(local=True, device=device)
image_classification = ImageClassification(local=True, device=device)

def test_vectorisation_single():
    out = vectorise("This is a sample thing.")
    assert isinstance(out, np.ndarray), "Not an array"
    assert type(out[0]) == np.dtype("float32"), "Vector dimension not float32"


def test_vectorisation_batch():
    out = vectorise(["This is a sample thing.",
                     "This is another sample thing."])
    assert isinstance(out, np.ndarray), "Not an array"
    assert len(out) == 2, "Not the right size"
    assert type(out[0][0]) == np.dtype(
        "float32"), "Vector dimension not float32"
    assert not np.array_equal(
        out[0], out[1]), "Two different texts have same vector"


def test_qa_single():
    out = qa(qa_example["q1"], qa_example["c"])
    assert type(out) == str, "Qa answer not a string"
    assert out == qa_example["a1"], "Wrong answer to simple qa"


def test_qa_batch():
    out = qa([qa_example["q1"], qa_example["q2"]],
             [qa_example["c"], qa_example["c"]])
    assert isinstance(out, list), "Output is not a list"
    assert len(out) == 2, "Incorrect number of answers"
    assert out[0] == qa_example["a1"], "Wrong answer to simple qa"
    assert out[1] == qa_example["a2"], "Wrong answer to simple qa"


def test_qa_conv_single():
    out = qa(qa_example["q2_conv"], qa_example["c"], prev_qa=[
             (qa_example["q1"], qa_example["a1"])])
    assert type(out) == str, "Qa answer not a string"
    assert out == qa_example["a2"], "Wrong answer to simple conversational qa"


def test_qa_conv_batch():
    questions = [qa_example["q2_conv"], qa_example["q3_conv"]]
    ctxs = [qa_example["c"], qa_example["c"]]
    prev_qa = [[(qa_example["q1"], qa_example["a1"])], [
        (qa_example["q2"], qa_example["a2"])]]

    out = qa(questions, ctxs, prev_qa=prev_qa)
    assert isinstance(out, list), "Output not a list"
    assert len(out) == 2, "Incorrect number of answers"
    assert out[0] == qa_example["a2"], "Wrong answer to simple conversational qa"
    assert out[1] == qa_example["a3"], "Wrong answer to simple conversational qa"


def test_summary_single():
    out = summarise(summary_context)
    assert type(out) == str, "Summary not a string"
    assert len(out) < len(summary_context), "Summary not shorter than input"


def test_summary_bulk():
    out = summarise([summary_context, summary_context])
    assert isinstance(out, list), "Output not a list"
    assert len(out) == 2, "Incorrect number of outputs"
    assert len(out[0]) < len(summary_context), "Summary not shorter than input"
    assert len(out[1]) < len(summary_context), "Summary not shorter than input"


def test_classify_single():
    out = classify(classify_context, classify_labels)
    assert isinstance(out, dict), "Output is not a dict"
    assert len(out.keys()) == len(
        classify_labels), "Incorrect number of labels"
    assert classify_correct == max(
        out, key=out.get), "Classification is severely wrong"


def test_classify_batch():
    out = classify([classify_context, classify_context],
                    [classify_labels, classify_labels])
    assert isinstance(out, list), "Output is not a list"

    for out in out:
        assert isinstance(out, dict), "List item is not a dict"
        assert len(out.keys()) == len(
            classify_labels), "Incorrect number of labels"
        assert classify_correct == max(
            out, key=out.get), "Classification is severely wrong"


def test_emotion_single():
    out = emotion("I am really angry.")
    assert type(out) == str, "Output is not a string"


def test_emotion_batch():
    out = emotion(["I am really angry.", "You are really angry."])
    assert isinstance(out, list), "Output is not a list"

    for out in out:
        assert type(out) == str, "List not made of strings"


def test_image_classification_single():
    out = image_classification(image_classification_image, image_classification_labels)
    assert isinstance(out, dict), "Output is not a dict"
    assert len(out.keys()) == len(
        image_classification_labels), "Incorrect number of labels"
    assert image_classification_correct == max(
        out, key=out.get), "Classification is severely wrong"


def test_image_classification_batch():
    out = image_classification([image_classification_image]*2, [image_classification_labels]*2)

    assert isinstance(out, list), "Output is not a list"

    for out in out:
        assert len(out.keys()) == len(
            image_classification_labels), "Incorrect number of labels"
        assert image_classification_correct == max(
            out, key=out.get), "Classification is severely wrong"
