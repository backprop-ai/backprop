.. _ft:

Finetuning
==========

Finetuning lets you take a model that has been trained on a very broad task and adapt it to your specific niche.

Finetuning is currently only supported for the T5 models (:ref:`t5` and :ref:`t5baseqasummaryemotion`).

The T5 models are text generation models that can take any text as input and produce any text as output.
This makes them very versatile for many tasks.

.. code-block:: python

    from kiri.models import T5
    from kiri.tasks import TextGeneration

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

See the in-depth `Finetuning <https://github.com/kiri-ai/kiri/blob/main/examples/core_functionality/Finetuning.ipynb>`_ example with code.