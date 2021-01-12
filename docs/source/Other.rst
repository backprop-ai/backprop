Kiri Core: Other Functionality
==============================

Q&A
---
Semantic search is a particularly valuable component for document-level Q&A.
Using the base search, Kiri can narrow down the most relevant documents for a given query.
From there, the document chunk that best suits the query can be determined to return an answer.

Q&A can also take a string directly, and answer questions from that provided context.

See the `Q&A <https://github.com/kiri-ai/kiri/blob/main/examples/core_functionality/Q%26A.ipynb>`_ examples with code.

Zero-Shot Classification
------------------------
Zero-shot classification is a relatively simple idea. 
As with standard classification, a model looks at input and assigns probabilities to a set of labels. However, with zero-shot, the model was not trained on any particular set of labels. 
This makes the classifier extremely flexible for a variety of use cases.

Check the example `zero-shot classification <https://github.com/kiri-ai/kiri/blob/main/examples/core_functionality/Classification.ipynb>`_ with code.

Sentiment Detection
-------------------
This is exactly what it says on the tin: analyzes emotional sentiment of some provided text input. 

Use is simple: just pass in a string of text, and get back an emotion or list of emotions.

See `sentiment detection <https://github.com/kiri-ai/kiri/blob/main/examples/core_functionality/Sentiment.ipynb>`_ with code.

Text Summarization
------------------
Also self-explanatory: takes a chunk of input text, and gives a summary of key information.

See the example for `text summarization <https://github.com/kiri-ai/kiri/blob/main/examples/core_functionality/Summary.ipynb>`_ with code.