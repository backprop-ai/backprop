.. _tasks:

Tasks
=====

Tasks are classes that act as a "middleman" between you and a model.
They know how to communicate with models running locally and in our API.

Supported Tasks
---------------

Tasks can be imported directly from ``backprop``.

See the full reference in :ref:`backprop-tasks`.

Q&A
^^^
Q&A answers a question based on a paragraph of text. It also supports previous questions and answers for a conversational setting.

.. code-block:: python

    from backprop import QA

    qa = QA()

    qa("Where does Sally live?", "Sally lives in London.")
    "London"

See the `Q&A <https://github.com/backprop-ai/backprop/blob/main/examples/Q%26A.ipynb>`_ examples with code.

Text Classification
^^^^^^^^^^^^^^^^^^^
Text Classification looks at input and assigns probabilities to a set of labels.

It is supported in 100+ languages: Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Basque, Belarusian, Bengali, Bengali Romanized, Bosnian, Breton, Bulgarian, Burmese, Burmese, Catalan, Chinese (Simplified), Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Hausa, Hebrew, Hindi, Hindi Romanized, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish (Kurmanji), Kyrgyz, Lao, Latin, Latvian, Lithuanian, Macedonian, Malagasy, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Sanskri, Scottish, Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tamil, Tamil Romanized, Telugu, Telugu Romanized, Thai, Turkish, Ukrainian, Urdu, Urdu Romanized, Uyghur, Uzbek, Vietnamese, Welsh, Western, Frisian, Xhosa, Yiddish.

.. code-block:: python

    from backprop import TextClassification

    tc = TextClassification()

    tc("I am mad because my product broke.", ["product issue", "nature"])
    {"product issue": 0.98, "nature": 0.05}

Check the example `text classification <https://github.com/backprop-ai/backprop/blob/main/examples/TextClassification.ipynb>`_ with code.

Sentiment Detection
^^^^^^^^^^^^^^^^^^^
This is exactly what it says on the tin: analyzes emotional sentiment of some provided text input. 

Use is simple: just pass in a string of text, and get back an emotion or list of emotions.

.. code-block:: python

    from backprop import Emotion

    emotion = Emotion()

    emotion("I really like what you did there")
    "approval"

See `sentiment detection <https://github.com/backprop-ai/backprop/blob/main/examples/Sentiment.ipynb>`_ with code.

Text Summarisation
^^^^^^^^^^^^^^^^^^
Also self-explanatory: takes a chunk of input text, and gives a summary of key information.

.. code-block:: python

    from backprop import Summarisation

    summarisation = Summarisation()

    summarisation("This is a long document that contains plenty of words")
    "short summary of document"

See the example for `text summarisation <https://github.com/backprop-ai/backprop/blob/main/examples/Summarisation.ipynb>`_ with code.

Image Classification
^^^^^^^^^^^^^^^^^^^^

Image classification functions exactly like text classification but for images.
It takes an image and a set of labels to calculate the probabilities for each label.

.. code-block:: python

    from backprop import ImageClassification

    ic = ImageClassification()

    ic("/home/Documents/dog.png", ["cat", "dog"])
    {"cat": 0.01, "dog": 0.99}

Check the example `image classification <https://github.com/backprop-ai/backprop/blob/main/examples/ImageClassification.ipynb>`_ with code.

Text Generation
^^^^^^^^^^^^^^^

Text Generation takes some text as input and generates more text based on it.

This is useful for story/idea generation or solving a broad range of tasks.

.. code-block:: python

    from backprop import TextGeneration

    tg = TextGeneration()

    tg("I like to go to")
    " the beach because I love the sun."

Check the example `text generation <https://github.com/backprop-ai/backprop/blob/main/examples/TextGeneration.ipynb>`_ with code.

Text Vectorisation
^^^^^^^^^^^^^^^^^^

Text Vectorisation takes some text and turns it into a vector.

This makes it possible to compare different texts numerically.
You could see how similar the vectors of two different paragraphs are, to group text automatically or build a semantic search engine.

.. code-block:: python

    from backprop import TextVectorisation

    tv = TextVectorisation()

    tv("iPhone 12 128GB")
    [0.92949192, 0.23123010, ...]