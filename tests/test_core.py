import pytest
from kiri import Kiri
import os

text = "Jon likes beer. He drinks it at pubs every Friday."

main_path = os.path.dirname(__file__)
image_classification_image = os.path.join(main_path, "data/dog.png")
image_classification_labels = ["cat", "dog"]
image_classification_correct = "dog"

kiri = Kiri(local=True, device="cpu")

def test_qa():
    out = kiri.qa("What does Jon like?", text)
    assert type(out) == str


def test_summarise():
    out = kiri.summarise(text)
    assert type(out) == str


def test_emotion():
    out = kiri.emotion(text)
    assert type(out) == str


def test_classify():
    out = kiri.classify(text, ["interests", "alcoholism"])
    assert isinstance(out, dict)
    out = kiri.classify_text(text, ["interests", "alcoholism"])
    assert isinstance(out, dict)


def test_classify():
    out = kiri.classify(text, ["interests", "alcoholism"])
    assert isinstance(out, dict)
    out = kiri.classify_text(text, ["interests", "alcoholism"])
    assert isinstance(out, dict)

def test_classify_image():
    out = kiri.classify_image(image_classification_image, image_classification_labels)
    assert isinstance(out, dict), "Output is not a dict"
    assert len(out.keys()) == len(
        image_classification_labels), "Incorrect number of labels"
    assert image_classification_correct == max(
        out, key=out.get), "Classification is severely wrong"
    
    out = kiri.image_classification(image_classification_image, image_classification_labels)
    assert isinstance(out, dict), "Output is not a dict"
    assert len(out.keys()) == len(
        image_classification_labels), "Incorrect number of labels"
    assert image_classification_correct == max(
        out, key=out.get), "Classification is severely wrong"

def test_generate_text():
    out = kiri.generate_text("This is something")
    assert type(out) == str, "Text Generation not a string"

    out = kiri.generate("This is something")
    assert type(out) == str, "Text Generation not a string"

def test_text_vectorisation():
    out = kiri.vectorise_text("This is a sample thing.")
    assert isinstance(out, list), "Not an array"
    assert type(out[0]) == float, "Vector dimension not float"

    out = kiri.vectorise("This is a sample thing.")
    assert isinstance(out, list), "Not an array"
    assert type(out[0]) == float, "Vector dimension not float"