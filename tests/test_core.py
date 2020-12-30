import pytest
from kiri import Kiri, Document, ChunkedDocument

text = "Jon likes beer. He drinks it at pubs every Friday."


def test_make_doc():
    c = "I am a document."
    doc = Document(c)
    assert doc.content == c


def test_make_chunk_doc():
    c = "I am a document. This is a sentence."
    doc = ChunkedDocument(c, chunking_level=1)
    assert doc.content == c
    assert doc.chunking_level == 1


def test_qa():
    kiri = Kiri()
    out = kiri.qa("What does Jon like?", text)
    assert type(out) == str


def test_summarise():
    kiri = Kiri()
    out = kiri.summarise(text)
    assert type(out) == str


def test_emotion():
    kiri = Kiri()
    out = kiri.emotion(text)
    assert type(out) == str


def test_classify():
    kiri = Kiri()
    out = kiri.classify(text, ["interests", "alcoholism"])
    assert isinstance(out, dict)


def test_doc_qa():
    c = "I am a document."
    doc = Document(c)
    out = doc.qa("What are you?")
    assert type(out) == str


def test_doc_summarise():
    c = "I am a document."
    doc = Document(c)
    out = doc.summarise()
    assert type(out) == str


def test_doc_summarise():
    c = "I am a document."
    doc = Document(c)
    out = doc.emotion()
    assert type(out) == str


def test_doc_summarise():
    c = "I am a document."
    doc = Document(c)
    out = doc.classify(["random"])
    assert isinstance(out, dict)
