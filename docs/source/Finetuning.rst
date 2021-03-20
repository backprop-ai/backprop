.. _ft:

Finetuning
==========

Finetuning lets you take a model that has been trained on a very broad task and adapt it to your specific niche.

Finetuning is currently supported for the following tasks and models:

* ``text-generation``:
    * :ref:`t5`
    * :ref:`t5baseqasummaryemotion`
  
* ``image-classification``:
    * :ref:`efficientnet`

The T5 models are text generation models that can take any text as input and produce any text as output.
This makes them very versatile for many tasks.

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

See the in-depth `Getting Started with Finetuning <https://github.com/backprop-ai/backprop/blob/main/examples/Finetuning.ipynb>`_ notebook with code.