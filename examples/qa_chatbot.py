from kiri import Kiri, ChunkedDocument, InMemoryDocStore
from big_n_docs import big_n_docs

"""
This example shows the use of Kiri's Q&A for a simple chatbot-esque loop.

In practice, Q&A performs search for document relevancy, and then
returns the top-rated answer within the documents found.
"""

elastic = False

if elastic:
    doc_store = ElasticDocStore("http://localhost:9000", doc_class=ElasticChunkedDocument, index="kiri_default")
    documents = big_n_docs["elastic"]
else:
    doc_store = InMemoryDocStore(doc_class=ChunkedDocument)
    documents = big_n_docs["memory"]

kiri = Kiri(doc_store)

kiri.upload(documents)

print("Hello! I'm a Kiri chatbot.")
# Hold previous Q/A pairs for additional context
session_qa = []
while True:
    try:
        question = input()
        answers = kiri.qa(question, prev_qa = session_qa)
        # Only showing the top-rated answer
        print(answers[0][0])
        prev_qa = (question, answers[0][0])
        session_qa += prev_qa            
        print()
    except Exception as e:
        print("Something broke, try again.")