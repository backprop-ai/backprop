Quickstart
==========

Installation
------------

Install Backprop via PyPi

.. code-block ::

    pip install backprop

Supported Tasks
---------------

Out of the box tasks you can solve with Backprop:

* Conversational question answering in English (for FAQ chatbots, text analysis, etc.)
* Text Classification in 100+ languages (for email sorting, intent detection, etc.)
* Image Classification (for object recognition, OCR, etc.)
* Text Vectorisation in 50+ languages (semantic search for ecommerce, documentation, etc.)
* Summarisation in English (TLDRs for long documents)
* Emotion detection in English (for customer satisfaction, text analysis, etc.)
* Text Generation (for idea, story generation and broad task solving)

For more specific use cases, you can adapt a task with little data and a couple of lines of code using finetuning. We are adding finetuning support for all tasks soon.

Basic Task Solving
------------------

.. code-block:: python

    import backprop

    context = "Take a look at the examples folder to see use cases!"

    qa = backprop.QA()

    # Start building!
    answer = qa("Where can I see what to build?", context)

    print(answer)
    # Prints
    "the examples folder"

See examples for all available tasks in :ref:`tasks`.

Basic Finetuning and Uploading
------------------------------

.. code-block:: python

    from backprop.models import T5
    from backprop import TextGeneration

    tg = TextGeneration(T5, local=True)

    # Any text works as training data
    inp = ["I really liked the service I received!", "Meh, it was not impressive."]
    out = ["positive", "negative"]

    # Finetune with a single line of code
    tg.finetune(inp, out)

    # Use your trained model
    prediction = tg("I enjoyed it!")

    print(prediction)
    # Prints
    "positive"

    # Upload to Backprop for production ready inference

    model = tg.model
    # Describe your model
    model.name = "t5-sentiment"
    model.description = "Predicts positive and negative sentiment"

    backprop.upload(model, api_key="abc")

Learn more about finetuning in :ref:`ft`.

Why Backprop?
-------------

1. No experience needed

   - Entrance to practical AI should be simple
   - Get state-of-the-art performance in your task without being an expert

2. Data is a bottleneck

   - Use AI without needing access to "big data"
   - With transfer learning, no data is required, but even a small amount can adapt a task to your niche.

3. There is an overwhelming amount of models

   - We implement the best ones for various tasks
   - A few general models can accomplish more with less optimisation

4. Deploying models cost effectively is hard work
   
   - If our models suit your use case, no deployment is needed
   - Adapt and deploy your own model with a couple of lines of code
   - Our API scales, is always available, and you only pay for usage