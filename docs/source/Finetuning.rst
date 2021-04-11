.. _ft:

Finetuning
==========

Finetuning lets you take a model that has been trained on a very broad task and adapt it to your specific niche.

Finetuning Parameters
^^^^^^^^^^^^^^^^^^^^^

There are a variety of parameters that can optionally be supplied when finetuning a task, to allow for more flexibility.

For references on each task's data input schema, find your task `here <tasks>`_.

* ``validation_split`` : Float value that determines the percentage of data that will be used for validation.
* ``epochs`` : Integer determining how many training iterations will be run while finetuning.
* ``batch_size`` : Integer specifying the batch size for training. Leavings this out lets Backprop determine it automatically for you.
* ``optimal_batch_size`` : Integer indicating the optimal batch size for the model to be used. This is model-specific, so in most cases
  will not need to be supplied.
* ``early_stopping_epochs`` : Integer value. When training, early stopping is a mechanism that determines a model has finished training based on
  lack of improvements to validation loss. This parameter indicates how many epochs will continue to run without seeing an improvement to validation loss.
  Default value is `1`.
* ``train_dataloader`` : DataLoader that will be used to pull batches from a dataset. We default this to be a DataLoader with the maximum number of workers
  (determined automatically by CPU). 
* ``val_dataloader`` : The same as ``train_dataloader``, for validation data.

Along with these parameters, finetuning has two keyword arguments that are functions, used for further customization.

* ``step`` : This function determines how a batch will be supplied to your chosen model, and returns loss. All of our included models/tasks have a default 
  ``step``, but for custom models, you can define exactly how to pass training data and calculate loss.
* ``configure_optimizers`` : Sets up an optimizer for use in training. As with ``step``, we include optimizers suited for each particular task. However,
  if you wish to experiment with other options, you can simply define a function that returns your chosen optimzer setup.

Basic Example
^^^^^^^^^^^^^

Here is a simple example of finetuning for text generation with T5.

.. code-block:: python

    from backprop.models import T5
    from backprop import TextGeneration

    tg = TextGeneration(T5)

    # Any text works as training data
    inp = ["I really liked the service I received!", "Meh, it was not impressive."]
    out = ["positive", "negative"]
    params = {"input_text": inp, "output_text": out}

    # Finetune with a single line of code
    tg.finetune(params)

    # Use your trained model
    prediction = tg("I enjoyed it!")


In-depth Example
^^^^^^^^^^^^^^^^    

See the in-depth `Getting Started with Finetuning <https://github.com/backprop-ai/backprop/blob/main/examples/Finetuning_GettingStarted.ipynb>`_ notebook with code.

Supported tasks and models
^^^^^^^^^^^^^^^^^^^^^^^^^^

Text Generation
---------------

References:

* :ref:`text-generation task <text-generation>`
* `text-generation finetuning notebook <https://github.com/backprop-ai/backprop/blob/main/examples/Finetuning_TextGeneration.ipynb>`_

Supported models:

* :ref:`T5 <t5>`
* :ref:`T5QASummaryEmotion <t5baseqasummaryemotion>`
  
Image Classification
--------------------

References:

* :ref:`image-classification task <image-classification>`
* `image-classification finetuning notebook <https://github.com/backprop-ai/backprop/blob/main/examples/Finetuning_ImageClassification.ipynb>`_

Supported models:

* :ref:`EfficientNet <efficientnet>`

Text Classification
-------------------

References:

* :ref:`text-classification task <text-classification>`
* `text-classification finetuning notebook <https://github.com/backprop-ai/backprop/blob/main/examples/Finetuning_TextClassification.ipynb>`_

Supported models:

* :ref:`XLNet <xlnet>`

Text Vectorisation
-------------------

References:

* :ref:`text-vectorisation task <text-vectorisation>`
* `text-vectorisation finetuning notebook <https://github.com/backprop-ai/backprop/blob/main/examples/Finetuning_TextVectorisation.ipynb>`_

Supported models:

* :ref:`DistiluseBaseMultilingualCasedV2 <distiluse>`
* :ref:`MSMARCODistilrobertaBaseV2 <distilroberta-msmarco>`


