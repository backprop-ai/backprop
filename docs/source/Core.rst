Core Functionality
==================

Using the Kiri class is the best way to use Kiri if you don't need to customise much.

.. code-block:: python

    from kiri import Kiri

    # Use our inference API
    k = Kiri(api_key="abc")
    # Or run locally
    k = Kiri(local=True)

The created instance can solve tasks locally or via the API.

You can also specify models to be used for the tasks. See the core reference for the options.

Supported Tasks
---------------

Q&A
^^^
Q&A answers a question based on a paragraph of text. It also supports previous questions and answers for a conversational setting.

.. code-block:: python

    k.qa("Where does Sally live?", "Sally lives in London.")
    "London"

See the `Q&A <https://github.com/kiri-ai/kiri/blob/main/examples/core_functionality/Q%26A.ipynb>`_ examples with code.

Text Classification
^^^^^^^^^^^^^^^^^^^
Text Classification looks at input and assigns probabilities to a set of labels.

It is supported in 100+ languages: Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Basque, Belarusian, Bengali, Bengali Romanized, Bosnian, Breton, Bulgarian, Burmese, Burmese, Catalan, Chinese (Simplified), Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Hausa, Hebrew, Hindi, Hindi Romanized, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish (Kurmanji), Kyrgyz, Lao, Latin, Latvian, Lithuanian, Macedonian, Malagasy, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Sanskri, Scottish, Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tamil, Tamil Romanized, Telugu, Telugu Romanized, Thai, Turkish, Ukrainian, Urdu, Urdu Romanized, Uyghur, Uzbek, Vietnamese, Welsh, Western, Frisian, Xhosa, Yiddish.

.. code-block:: python

    k.classify_text("I am mad because my product broke.", ["product issue", "nature"])
    {"product issue": 0.98, "nature": 0.05}

Check the example `text classification <https://github.com/kiri-ai/kiri/blob/main/examples/core_functionality/TextClassification.ipynb>`_ with code.

Sentiment Detection
^^^^^^^^^^^^^^^^^^^
This is exactly what it says on the tin: analyzes emotional sentiment of some provided text input. 

Use is simple: just pass in a string of text, and get back an emotion or list of emotions.

.. code-block:: python

    k.emotion("I really like what you did there")
    "approval"

See `sentiment detection <https://github.com/kiri-ai/kiri/blob/main/examples/core_functionality/Sentiment.ipynb>`_ with code.

Text Summarisation
^^^^^^^^^^^^^^^^^^
Also self-explanatory: takes a chunk of input text, and gives a summary of key information.

See the example for `text summarisation <https://github.com/kiri-ai/kiri/blob/main/examples/core_functionality/Summarisation.ipynb>`_ with code.

.. code-block:: python

    k.summarise("This is a long document that contains plenty of words")
    "short summary of document"

Image Classification
^^^^^^^^^^^^^^^^^^^^

Image classification functions exactly like text classification but for images.
It takes an image and a set of labels to calculate the probabilities for each label.

.. code-block:: python

    k.image_classification("/home/Documents/dog.png", ["cat", "dog"])
    {"cat": 0.01, "dog": 0.99}

Check the example `image classification <https://github.com/kiri-ai/kiri/blob/main/examples/core_functionality/ImageClassification.ipynb>`_ with code.

Text Generation
^^^^^^^^^^^^^^^

Text Generation takes some text as input and generates more text based on it.

This is useful for story/idea generation or solving a broad range of tasks.

.. code-block:: python

    k.generate_text("I like to go to")
    " the beach because I love the sun."

Check the example `text generation <https://github.com/kiri-ai/kiri/blob/main/examples/core_functionality/TextGeneration.ipynb>`_ with code.

Text Vectorisation
^^^^^^^^^^^^^^^^^^

Text Vectorisation takes some text and turns it into a vector.

This makes it possible to compare different texts numerically.
You could see how similar the vectors of two different paragraphs are, to group text automatically or build a semantic search engine.

.. code-block:: python

    k.vectorise_text("iPhone 12 128GB")
    [0.92949192, 0.23123010, ...]


Supported Utility Methods
-------------------------

.. code-block:: python

    # Saves model instance to ~/.cache/kiri/model_name
    # model_name is determined from model_instance.name
    k.save(model_instance)

    # Loads model instance from ~/.cache/kiri/model_name
    model = k.load("model_name")

    # Uploads model to Kiri for production ready inference
    k.upload(model, api_key="abc")