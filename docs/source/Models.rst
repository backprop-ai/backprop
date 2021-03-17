.. _models:

Models
======

Models are classes that power tasks. You can import models from ``backprop.models``.

A valid model must: 

* implement one or more tasks according to the task's input schema
* have a valid list of tasks (strings) as a tasks field
* have a name field that must be between 3 and 100 lowercase a-z characters, with numbers, dashes (-) and underscores (\_) allowed.
* produce JSON serializable output
* not have any unspecified external dependencies
* be pickleable with ``dill``

If these criteria are fulfilled, then it is very easy to save, load and upload the model.

A model can:

* offer finetuning support via the ``finetune`` method
* support single and batched task requests

Every model included in our library fulfils the necessary criteria.
Our models also support both single and batched requests.
For example, you can vectorise text with both ``"some text"`` and ``["first text", "second text"]`` as input. 

We are still working on adding finetuning support to at least one model for each task.

See the full model reference in :ref:`backprop-models`.

Generic Models
--------------

Generic models are used to implement more specific models. Generic models don't support any tasks out of the box.
When implementing a model, it is useful to inherit from a generic model to ensure it fits in with other Backprop modules.

Example usage:

.. code-block:: python

    from backprop.models import PathModel

    from transformers import AutoModelForPreTraining, AutoTokenizer

    model = PathModel("t5-base", init_model=AutoModelForPreTraining, init_tokenizer=AutoTokenizer)

    # Use model
    model([0, 1, 2])

See an example how to implement a generic model for a task.

See the generic models reference in :ref:`backprop-models`.

Custom Models
-------------

Custom models, or just models, are classes that have been adapted and optimised to solve tasks.

Example usage:

.. code-block:: python

    from backprop.models import T5QASummaryEmotion

    model = T5QASummaryEmotion()

    # Use model
    model({"text": "This is pretty cool!"}, task="emotion")
    "admiration"

This is an ever-growing list, so check out the full reference in :ref:`backprop-models`.

Supporting Tasks with Models
----------------------------

In order for a model to support a task, it must follow the task's input schema.

For each task, the input consists of an argument and a keyword argument.

This is passed by calling the model object directly.

.. code-block:: python

    from backprop.models import BaseModel

    class MyModel(BaseModel):
        def __call__(self, task_input, task="emotion"):
            if task == "emotion":
                text = task_input.get("text")
                # Do some AI magic with text, assume result is "admiration"
                return "admiration"
            else:
                raise ValueError("Unsupported task!")
    
    
    model = MyModel()

    # Use model
    model({"text": "This is pretty cool!"}, task="emotion")
    "admiration"

The input argument is a dictionary, while the keyword argument ``task`` is a string.

Q&A
^^^

Task string is ``"qa"``.

Dictionary argument specification:

+----------+--------------------------------------+---------------------------------------------------------------+
| key      | type                                 | description                                                   |
+==========+======================================+===============================================================+
| question | ``str`` or ``List[str]``             | question or list of questions                                 |
+----------+--------------------------------------+---------------------------------------------------------------+
| context  | ``str`` or ``List[str]``             | context or list of contexts                                   |
+----------+--------------------------------------+---------------------------------------------------------------+
| prev_q   | ``List[str]`` or ``List[List[str]]`` | List of previous questions or list of previous question lists |
+----------+--------------------------------------+---------------------------------------------------------------+
| prev_a   | ``List[str]`` or                     | List of previous answers or list of previous answer lists     |
|          | ``List[List[str]]``                  |                                                               |
+----------+--------------------------------------+---------------------------------------------------------------+

Text Classification
^^^^^^^^^^^^^^^^^^^
Task string is ``"text-classification"``.

Dictionary argument specification:

+--------+--------------------------------------+-----------------------------------------------------+
| key    | type                                 | description                                         |
+========+======================================+=====================================================+
| text   | ``str`` or ``List[str]``             | text or list of texts to classify                   |
+--------+--------------------------------------+-----------------------------------------------------+
| labels | ``List[str]`` or ``List[List[str]]`` | labels or list of labels to assign probabilities to |
+--------+--------------------------------------+-----------------------------------------------------+

Sentiment Detection (Emotion)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Task string is ``"emotion"``.

+------+--------------------------+----------------------------------------------+
| key  | type                     | description                                  |
+======+==========================+==============================================+
| text | ``str`` or ``List[str]`` | text or list of texts to detect emotion from |
+------+--------------------------+----------------------------------------------+

Text Summarisation
^^^^^^^^^^^^^^^^^^

Task string is ``"summarisation"``.

+------+--------------------------+------------------------------------+
| key  | type                     | description                        |
+======+==========================+====================================+
| text | ``str`` or ``List[str]`` | text or list of texts to summarise |
+------+--------------------------+------------------------------------+

Image Classification
^^^^^^^^^^^^^^^^^^^^

Task string is ``"image-classification"``.

+--------+--------------------------------------+-------------------------------------------------------+
| key    | type                                 | description                                           |
+========+======================================+=======================================================+
| image  | ``str`` or ``List[str]``             | base64 encoded image or list of base64 encoded images |
+--------+--------------------------------------+-------------------------------------------------------+
| labels | ``List[str]`` or ``List[List[str]]`` | labels or list of labels to assign probabilities to   |
+--------+--------------------------------------+-------------------------------------------------------+

Text Generation
^^^^^^^^^^^^^^^

Task string is ``"text-generation"``.

+--------------------+--------------------------+----------------------------------------------------------------------------------------------------------------------+
| key                | type                     | description                                                                                                          |
+====================+==========================+======================================================================================================================+
| text               | ``str`` or ``List[str]`` | text or list of texts to generate from                                                                               |
+--------------------+--------------------------+----------------------------------------------------------------------------------------------------------------------+
| min_length         | ``int``                  | minimum number of tokens to generate                                                                                 |
+--------------------+--------------------------+----------------------------------------------------------------------------------------------------------------------+
| max_length         | ``int``                  | maximum number of tokens to generate                                                                                 |
+--------------------+--------------------------+----------------------------------------------------------------------------------------------------------------------+
| temperature        | ``float``                | value that alters softmax probabilities                                                                              |
+--------------------+--------------------------+----------------------------------------------------------------------------------------------------------------------+
| top_k              | ``float``                | sampling strategy in which probabilities are redistributed among top k most-likely words                             |
+--------------------+--------------------------+----------------------------------------------------------------------------------------------------------------------+
| top_p              | ``float``                | sampling strategy in which probabilities are distributed among set of words with combined probability greater than p |
+--------------------+--------------------------+----------------------------------------------------------------------------------------------------------------------+
| repetition_penalty | ``float``                | penalty to be applied to words present in the text and words already generated in the sequence                       |
+--------------------+--------------------------+----------------------------------------------------------------------------------------------------------------------+
| length_penalty     | ``float``                | penalty applied to overall sequence length. >1 for longer sequences, or <1 for shorter ones                          |
+--------------------+--------------------------+----------------------------------------------------------------------------------------------------------------------+
| num_beams          | ``int``                  | number of beams to be used in beam search                                                                            |
+--------------------+--------------------------+----------------------------------------------------------------------------------------------------------------------+
| num_generations    | ``int``                  | number of times to generate                                                                                          |
+--------------------+--------------------------+----------------------------------------------------------------------------------------------------------------------+
| do_sample          | ``bool``                 | whether to sample or do greedy search                                                                                |
+--------------------+--------------------------+----------------------------------------------------------------------------------------------------------------------+

Text Vectorisation
^^^^^^^^^^^^^^^^^^

Task string is ``"text-vectorisation"``.

+------+--------------------------+------------------------------------+
| key  | type                     | description                        |
+======+==========================+====================================+
| text | ``str`` or ``List[str]`` | text or list of texts to vectorise |
+------+--------------------------+------------------------------------+