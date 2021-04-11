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

Inference
*********

.. code-block:: python

    from backprop import QA

    qa = QA()

    qa("Where does Sally live?", "Sally lives in London.")
    "London"

Finetuning
**********
The params dictionary for Q&A training consists of contexts, questions, answers, and optionally a list of previous Q/A pairs to be used in context.

The ``prev_qa`` parameter is used to make a Q&A system conversational -- that is to say, it can infer the meaning of words in a new question, based on previous questions.

Finetuning Q&A also accepts keyword arguments ``max_input_length`` and ``max_output_length``, which specify the maximum token length for inputs and outputs.

.. code-block:: python
    import backprop
                
    # Initialise task
    qa = backprop.QA()

    # Set up training data for QA. Note that repeated contexts are needed, along with empty prev_qas to match.
    # Input must be completely 1:1, each question has an associated answer, context, and prev_qa (if prev_qa is to be used).

    questions = ["What's Backprop?", "What language is it in?", "When was the Moog synthesizer invented?"]
    answers = ["A library that trains models", "Python", "1964"]
    contexts = ["Backprop is a Python library that makes training and using models easier.", 
                "Backprop is a Python library that makes training and using models easier.",
                "Bob Moog was a physicist. He invented the Moog synthesizer in 1964."]

    prev_qas = [[], 
                [("What's Backprop?", "A library that trains models")],
                []]

    params = {"questions": questions,
            "answers": answers,
            "contexts": contexts,
            "prev_qas": prev_qas}

    # Finetune
    qa.finetune(params=params)

Consider the second example in the training data above. The question "What language is it in?" does not mean much on its own. But because the model can access the 
matching ``prev_qa`` example, it gets additional information: as the first question reads, "What's Backprop?", it can infer that the "it" in the next question refers to Backprop as well.

If you use ``prev_qa``, you must ensure it is 1:1 with the rest of your input data: even if a row does not have any previous question/answer pairs, an empty list is still required at that
index in ``prev_qa``. 

See the `Q&A Notebook <https://github.com/backprop-ai/backprop/blob/main/examples/Q%26A.ipynb>`_ examples with code.

See the :ref:`Q&A Task Reference <qa>`.

Text Classification
^^^^^^^^^^^^^^^^^^^
Text Classification looks at input and assigns probabilities to a set of labels.

It is supported in 100+ languages: Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Basque, Belarusian, Bengali, Bengali Romanized, Bosnian, Breton, Bulgarian, Burmese, Burmese, Catalan, Chinese (Simplified), Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Hausa, Hebrew, Hindi, Hindi Romanized, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish (Kurmanji), Kyrgyz, Lao, Latin, Latvian, Lithuanian, Macedonian, Malagasy, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Sanskri, Scottish, Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tamil, Tamil Romanized, Telugu, Telugu Romanized, Thai, Turkish, Ukrainian, Urdu, Urdu Romanized, Uyghur, Uzbek, Vietnamese, Welsh, Western, Frisian, Xhosa, Yiddish.

Inference
*********

.. code-block:: python

    from backprop import TextClassification

    tc = TextClassification()

    tc("I am mad because my product broke.", ["product issue", "nature"])
    {"product issue": 0.98, "nature": 0.05}

Finetuning
**********
Supplying parameters for text classification is straightforward: the params dict contains the keys "texts" and "labels".
The values of these keys are lists of input texts and the labels to which they are assigned. 
When you finetune, Backprop will automatically set up a model with the correct number of outputs (based on the unique labels passed in).

Finetuning text classification also accepts the keyword argument ``max_length``, which specifoes the maximum token length for inputs.

.. code-block:: python
    import backprop

    tc = backprop.TextCLassification()

    # Set up input data. Labels will automatically be used to set up model with number of classes for classification.
    inp = ["This is a political news article", "This is a computer science research paper", "This is a movie review"]
    out = ["Politics", "Science", "Entertainment"]
    params = {"texts": inp, "labels": out}

    # Finetune
    tc.finetune(params)

Check the example `Text Classification Notebook <https://github.com/backprop-ai/backprop/blob/main/examples/TextClassification.ipynb>`_ with code.

See the :ref:`Text Classification Task Reference <text-classification>`.

Sentiment/Emotion Detection
^^^^^^^^^^^^^^^^^^^
This is exactly what it says on the tin: analyzes emotional sentiment of some provided text input. 

Inference
*********

Use is simple: just pass in a string of text, and get back an emotion or list of emotions.

.. code-block:: python

    from backprop import Emotion

    emotion = Emotion()

    emotion("I really like what you did there")
    "approval"

Finetuning
**********
Sentiment detection finetuning is currently a generative task. This will likely be converted to a wrapper around Text Classification in the future.

The schema will remain the same, however: the emotion task params dict contains the keys "input_text" and "output_text".
The inputs are the strings to be analysed, and the outputs are the emotions corresponding to those inputs.

Finetuning this task also accepts keyword arguments ``max_input_length`` and ``max_output_length``, which specify the maximum token length for inputs and outputs.

.. code-block:: python
    import backprop
            
    emote = backprop.Emotion()

    # Provide sentiment data for training
    inp = ["I really liked the service I received!", "Meh, it was not impressive."]
    out = ["positive", "negative"]
    params = {"input_text": inp, "output_text": out}

    # Finetune
    emote.finetune(params)

See `Sentiment Detection Notebook <https://github.com/backprop-ai/backprop/blob/main/examples/Sentiment.ipynb>`_ with code.

See the :ref:`Emotion Task Reference <emotion>`.

Text Summarisation
^^^^^^^^^^^^^^^^^^
Also self-explanatory: takes a chunk of input text, and gives a summary of key information.

Inference
*********

.. code-block:: python

    from backprop import Summarisation

    summarisation = Summarisation()

    summarisation("This is a long document that contains plenty of words")
    "short summary of document"

Finetuning
**********
The summarisation input schema is a params dict with "input_text" and "output_text" keys. Inputs would be longer pieces of text, and the corresponding outputs are
summarised versions of the same text.

Finetuning sumamrisation also accepts keyword arguments ``max_input_length`` and ``max_output_length``, which specify the maximum token length for inputs and outputs.

.. code-block:: python

    import backprop

    summary = backprop.Summarisation()

    # Provide training data for task
    inp = ["This is a long news article about recent political happenings.", "This is an article about some recent scientific research."]
    out = ["Short political summary.", "Short scientific summary."]
    params = {"input_text": inp, "output_text": out}

    # Finetune
    summary.finetune(params)

See the example for `Text Summarisation Notebook <https://github.com/backprop-ai/backprop/blob/main/examples/Summarisation.ipynb>`_ with code.

See the :ref:`Text Summarisation Task Reference <summarisation>`.

Image Classification
^^^^^^^^^^^^^^^^^^^^

Image classification functions exactly like text classification but for images.
It takes an image and a set of labels to calculate the probabilities for each label.

Inference
*********

.. code-block:: python

    from backprop import ImageClassification

    ic = ImageClassification()

    ic("/home/Documents/dog.png", ["cat", "dog"])
    {"cat": 0.01, "dog": 0.99}

Finetuning
**********
The params dict for image classification consists of "images" (input images) and "labels" (image labels).
This task also includes variants for single-label and multi-label classification.

.. code-block:: python

    import backprop

    ic = backprop.ImageClassification()

    # Prep training images/labels. Labels are automatically used to set up model with number of classes for classification.
    images = ["images/beagle/photo.jpg", "images/dachsund/photo.jpg", "images/malamute/photo.jpg"]
    labels = ["beagle", "dachsund", "malamute"]
    params = {"images": images, "labels": labels}

    # Finetune
    ic.finetune(params, variant="single_label")

Check the example `Image Classification Notebook <https://github.com/backprop-ai/backprop/blob/main/examples/ImageClassification.ipynb>`_ with code.

See the :ref:`Image Classification Task Reference <image-classification>`.

Image Vectorisation
^^^^^^^^^^^^^^^^^^^

Image Vectorisation takes an image and turns it into a vector.

This makes it possible to compare different images numerically.

Inference
*********

.. code-block:: python

    from backprop import ImageVectorisation

    iv = ImageVectorisation()

    iv("/home/Documents/dog.png")
    [0.92949192, 0.23123010, ...]

Finetuning
**********
When finetuning image vectorisation, the task input determines on the loss variant you plan to use.
This comes in two flavors: triplet, or cosine similarity.

The default is triplet. This schema requires keys "images" (input images), and "groups" (group in which each image falls). This variant uses a distinct sampling strategy,
based on group numbers. A given "anchor" image is compared to a positive match (same group number) and a negative match (different group number). The goal is to minimise the
distance between the anchor vector and the positive match vector, while also maximising the distance between the anchor vector and negative match vector.

For cosine similarity, the schema is different. It requires keys "imgs1", "imgs2", and "similarity_scores". When training on row *x*, this variant
vectorises `imgs1[x]` and `imgs2[x]`, with the target cosine similarity being the value at `similarity_scores[x]`.

.. code-block:: python

    import backprop

    iv = backprop.ImageVectorisation()

    # Set up training data & finetune (triplet variant)
    images = ["images/beagle/photo.jpg",  "images/shiba_inu/photo.jpg", "images/beagle/photo1.jpg", "images/malamute/photo.jpg"]
    groups = [0, 1, 0, 2]
    params = {"images": images, "groups": groups}

    iv.finetune(params, variant="triplet")

    # Set up training data & finetune (cosine_similarity variant)
    imgs1 = ["images/beagle/photo.jpg", "images/shiba_inu/photo.jpg"]
    imgs2 = ["images/beagle/photo1.jpg", "images/malamute/photo.jpg"]
    similarity_scores = [1.0, 0.0]
    params = {"imgs1": imgs1, "imgs2": imgs2, "similarity_scores": similarity_scores}

    iv.finetune(params, variant="cosine_similarity")


Check the example `Image Vectorisation Notebook <https://github.com/backprop-ai/backprop/blob/main/examples/ImageVectorisation.ipynb>`_ with code.

See the :ref:`Image Vectorisation Task Reference <image-vectorisation>`.

Text Generation
^^^^^^^^^^^^^^^

Text Generation takes some text as input and generates more text based on it.

This is useful for story/idea generation or solving a broad range of tasks.

Inference
*********

.. code-block:: python

    from backprop import TextGeneration

    tg = TextGeneration()

    tg("I like to go to")
    " the beach because I love the sun."

Finetuning
**********
Text generation requires a params dict with keys "input_text" and "output_text". The values here are simply lists of strings.

When trained, the model will learn expected outputs for a given context -- this is how tasks such as generative sentiment detection or text summary can be trained.

Finetuning text generation also accepts keyword arguments ``max_input_length`` and ``max_output_length``, which specify the maximum token length for inputs and outputs.

.. code-block:: python

    import backprop
            
    tg = backprop.TextGeneration()

    # Any text works as training data
    inp = ["I really liked the service I received!", "Meh, it was not impressive."]
    out = ["positive", "negative"]
    params = {"input_text": inp, "output_text": out}

    # Finetune
    tg.finetune(params)

Check the example `Text Generation Notebook <https://github.com/backprop-ai/backprop/blob/main/examples/TextGeneration.ipynb>`_ with code.

See the :ref:`Text Generation Task Reference <text-generation>`.

Text Vectorisation
^^^^^^^^^^^^^^^^^^

Text Vectorisation takes some text and turns it into a vector.

This makes it possible to compare different texts numerically.
You could see how similar the vectors of two different paragraphs are, to group text automatically or build a semantic search engine.

Inference
*********

.. code-block:: python

    from backprop import TextVectorisation

    tv = TextVectorisation()

    tv("iPhone 12 128GB")
    [0.92949192, 0.23123010, ...]

Finetuning
**********
When finetuning text vectorisation, the task input determines on the loss variant you plan to use.
Like with image vectorisation, this can be either "triplet" or "cosine_similarity".

The default is cosine_similarity. It requires keys "texts1", "texts2", and "similarity_scores". When training on row *x*, this variant
vectorises `texts1[x]` and `texts2[x]`, with the target cosine similarity being the value at `similarity_scores[x]`.

Triplet is different. This schema requires keys "texts" (input texts), and "groups" (group in which each piece of text falls). This variant uses a distinct sampling strategy,
based on group numbers. A given "anchor" text is compared to a positive match (same group number) and a negative match (different group number). The goal is to minimise the
distance between the anchor vector and the positive match vector, while also maximising the distance between the anchor vector and negative match vector.


Finetuning text vectorisation also accepts the keyword argument ``max_length`` which specifies the maximum token length for encoded text.

.. code-block:: python
    
    import backprop

    tv = backprop.TextVectorisation()

    # Set up training data & finetune (cosine_similarity variant)
    texts1 = ["I went to the store and bought some bread", "I am getting a cat soon"]
    texts2 = ["I bought bread from the store", "I took my dog for a walk"]
    similarity_scores = [1.0, 0.0]
    params = {"texts1": texts1, "texts2": texts2, "similarity_scores": similarity_scores}

    tv.finetune(params, variant="cosine_similarity")

    # Set up training data & finetune (triplet variant)
    texts = ["I went to the store and bought some bread", "I bought bread from the store", "I'm going to go walk my dog"]
    groups = [0, 0, 1]
    params = {"texts": texts, "groups": groups}

    tv.finetune(params, variant="triplet")

Check the example `Text Vectorisation Notebook <https://github.com/backprop-ai/backprop/blob/main/examples/TextVectorisation.ipynb>`_ with code.

See the :ref:`Text Vectorisation Task Reference <text-vectorisation>`.