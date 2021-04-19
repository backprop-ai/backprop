.. _models:

Models
======

Models are classes that power tasks. 99% of users should not have to use models directly.
Instead, you should use the appropriate :ref:`tasks` where you specify the model.

See the available models and tasks they support in the Backprop `Model Hub<https://backprop.co/hub>`_.

The only valid reason for using models directly is if you are implementing your own.

You can import models from ``backprop.models``.

See the full model reference in :ref:`backprop-models`.

Model Requirements
------------------

A valid model must: 

* implement one or more tasks according to the task's input schema
* have a valid list of tasks (strings) as a tasks field
* have a name field that must be between 3 and 100 lowercase a-z characters, with numbers, dashes (-) and underscores (\_) allowed.
* produce JSON serializable output
* not have any unspecified external dependencies
* be pickleable with ``dill``

If these criteria are fulfilled, then it is very easy to save, load and upload the model.

A model can:

* offer finetuning support by implementing the ``process_batch``, ``training_step`` and optionally ``pre_finetuning`` methods
* support single and batched task requests

Every model included in our library fulfils the necessary criteria.
Our models also support both single and batched requests.
For example, you can vectorise text with both ``"some text"`` and ``["first text", "second text"]`` as input. 

Auto Model
----------

``AutoModel`` is a special class that can load any supported model by its string identifier.
You can also use it to see what models are available.

Example usage:

.. code-block:: python

    from backprop.models import AutoModel

    AutoModel.list_models(display=True, limit=10)

    model = AutoModel.from_pretrained("distilgpt2")

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


Supporting Task inference with Models
-------------------------------------

In order for a model to support inference for a task, it must follow the task's input schema.

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

+--------+--------------------------------------+--------------------------------------------------------------------------+
| key    | type                                 | description                                                              |
+========+======================================+==========================================================================+
| text   | ``str`` or ``List[str]``             | text or list of texts to classify                                        |
+--------+--------------------------------------+--------------------------------------------------------------------------+
| labels | ``List[str]`` or ``List[List[str]]`` | optional (zero-shot) labels or list of labels to assign probabilities to |
+--------+--------------------------------------+--------------------------------------------------------------------------+
| top_k  | ``int``                              | optional number of highest probability labels to return                  |
+--------+--------------------------------------+--------------------------------------------------------------------------+

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

+--------+------------------------------------------------------------------+--------------------------------------------------------------------------+
| key    | type                                                             | description                                                              |
+========+==================================================================+==========================================================================+
| image  | ``str`` or ``List[str]`` or ``PIL.Image`` or ``List[PIL.Image]`` | PIL or base64 encoded image or list of them                              |
+--------+------------------------------------------------------------------+--------------------------------------------------------------------------+
| labels | ``List[str]`` or ``List[List[str]]``                             | optional (zero-shot) labels or list of labels to assign probabilities to |
+--------+------------------------------------------------------------------+--------------------------------------------------------------------------+
| top_k  | ``int``                                                          | optional number of highest probability labels to return                  |
+--------+------------------------------------------------------------------+--------------------------------------------------------------------------+

Image Vectorisation
^^^^^^^^^^^^^^^^^^^

Task string is ``"image-vectorisation"``.

+-------+------------------------------------------------------------------+---------------------------------------------+
| key   | type                                                             | description                                 |
+=======+==================================================================+=============================================+
| image | ``str`` or ``List[str]`` or ``PIL.Image`` or ``List[PIL.Image]`` | PIL or base64 encoded image or list of them |
+-------+------------------------------------------------------------------+---------------------------------------------+

Image-Text Vectorisation
^^^^^^^^^^^^^^^^^^^

Task string is ``"image-text-vectorisation"``.

+-------+------------------------------------------------------------------+---------------------------------------------+
| key   | type                                                             | description                                 |
+=======+==================================================================+=============================================+
| image | ``str`` or ``List[str]`` or ``PIL.Image`` or ``List[PIL.Image]`` | PIL or base64 encoded image or list of them |
+-------+------------------------------------------------------------------+---------------------------------------------+
| text  | ``str`` or ``List[str]``                                         | text or list of texts to vectorise          |
+-------+------------------------------------------------------------------+---------------------------------------------+

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

Supporting Task finetuning with Models
--------------------------------------

In order for a model to support finetuning for a task, it must follow the task's finetuning schema.

This involves implementing three methods:

1. ``process_batch`` - receive task specific data and process it
2. ``training_step`` - receive data processed by the ``process_batch`` method and produce output
3. ``pre_finetuning`` - optionally receive task specific parameters and adjust the model before finetuning

The inputs and outputs for each of these methods vary depending on the task.

Q&A
^^^

``process_batch`` takes dictionary argument ``params`` and keyword argument ``task="qa"``.

``params`` has the following keys and values:

+-------------------+---------------------------+----------------------------------------+
| key               | type                      | description                            |
+===================+===========================+========================================+
| question          | ``str``                   | Question                               |
+-------------------+---------------------------+----------------------------------------+
| context           | ``str``                   | Context that contains answer           |
+-------------------+---------------------------+----------------------------------------+
| prev_qa           | ``List[Tuple[str, str]]`` | List of previous question-answer pairs |
+-------------------+---------------------------+----------------------------------------+
| output            | ``str``                   | Answer                                 |
+-------------------+---------------------------+----------------------------------------+
| max_input_length  | ``int``                   | Max number of tokens in input          |
+-------------------+---------------------------+----------------------------------------+
| max_output_length | ``int``                   | Max number of tokens in output         |
+-------------------+---------------------------+----------------------------------------+

``training_step`` must return loss.

``pre_finetuning`` is not used.

Text Classification
^^^^^^^^^^^^^^^^^^^

Currently, only the single label variant is supported.

``process_batch`` takes dictionary argument ``params`` and keyword argument ``task="text-classification"``.

``params`` has the following keys and values:

+--------------+---------+--------------------------------+
| key          | type    | description                    |
+==============+=========+================================+
| inputs       | ``str`` | Text                           |
+--------------+---------+--------------------------------+
| class_to_idx | ``str`` | Maps labels to integers        |
+--------------+---------+--------------------------------+
| labels       | ``str`` | Correct label                  |
+--------------+---------+--------------------------------+
| max_length   | ``str`` | Max number of tokens in inputs |
+--------------+---------+--------------------------------+

``training_step`` must return loss.

``pre_finetuning`` takes labels argument which is a dictionary that maps integers (from 0 to n) to labels.

Sentiment Detection (Emotion)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``process_batch`` takes dictionary argument ``params`` and keyword argument ``task="emotion"``.

``params`` has the following keys and values:

+-------------------+---------+--------------------------------+
| key               | type    | description                    |
+===================+=========+================================+
| input             | ``str`` | Text to detect emotion from    |
+-------------------+---------+--------------------------------+
| output            | ``str`` | Emotion text                   |
+-------------------+---------+--------------------------------+
| max_input_length  | ``int`` | Max number of tokens in input  |
+-------------------+---------+--------------------------------+
| max_output_length | ``int`` | Max number of tokens in output |
+-------------------+---------+--------------------------------+

``training_step`` must return loss.

``pre_finetuning`` is not used.

Text Summarisation
^^^^^^^^^^^^^^^^^^

``process_batch`` takes dictionary argument ``params`` and keyword argument ``task="summarisation"``.

``params`` has the following keys and values:

+-------------------+---------+--------------------------------+
| key               | type    | description                    |
+===================+=========+================================+
| input             | ``str`` | Text to summarise              |
+-------------------+---------+--------------------------------+
| output            | ``str`` | Summary                        |
+-------------------+---------+--------------------------------+
| max_input_length  | ``int`` | Max number of tokens in input  |
+-------------------+---------+--------------------------------+
| max_output_length | ``int`` | Max number of tokens in output |
+-------------------+---------+--------------------------------+

``training_step`` must return loss.

``pre_finetuning`` is not used.

Image Classification
^^^^^^^^^^^^^^^^^^^^

``process_batch`` takes dictionary argument ``params`` and keyword argument ``task="image-classification"``.

``params`` has the following keys and values:

+-------+---------+---------------+
| key   | type    | description   |
+=======+=========+===============+
| image | ``str`` | Path to image |
+-------+---------+---------------+

``training_step`` must return logits for each class (label).

``pre_finetuning`` takes:

* ``labels`` keyword argument which is a dictionary that maps integers (from 0 to n) to labels.
* ``num_classes`` keyword argument which is an integer for the number of unique labels.

Image Vectorisation
^^^^^^^^^^^^^^^^^^^

``process_batch`` takes dictionary argument ``params`` and keyword argument ``task="image-vectorisation"``.

``params`` has the following keys and values:

+-------+---------+---------------+
| key   | type    | description   |
+=======+=========+===============+
| image | ``str`` | Path to image |
+-------+---------+---------------+

``training_step`` must return vector tensor.

``pre_finetuning`` takes no arguments.

Text Generation
^^^^^^^^^^^^^^^

``process_batch`` takes dictionary argument ``params`` and keyword argument ``task="text-generation"``.

``params`` has the following keys and values:

+-------------------+---------+--------------------------------+
| key               | type    | description                    |
+===================+=========+================================+
| input             | ``str`` | Generation prompt              |
+-------------------+---------+--------------------------------+
| output            | ``str`` | Generation outpu               |
+-------------------+---------+--------------------------------+
| max_input_length  | ``int`` | Max number of tokens in input  |
+-------------------+---------+--------------------------------+
| max_output_length | ``int`` | Max number of tokens in output |
+-------------------+---------+--------------------------------+

``training_step`` must return loss.

``pre_finetuning`` is not used.

Text Vectorisation
^^^^^^^^^^^^^^^^^^

``process_batch`` takes dictionary argument ``params`` and keyword argument ``task="text-vectorisation"``.

``params`` has the following keys and values:

+------+---------+-------------------+
| key  | type    | description       |
+======+=========+===================+
| text | ``str`` | Text to vectorise |
+------+---------+-------------------+

``training_step`` must return vector tensor.

``pre_finetuning`` takes no arguments.

Image-Text Vectorisation
^^^^^^^^^^^^^^^^^^^^^^^^

``process_batch`` takes dictionary argument ``params`` and keyword argument ``task="image-text-vectorisation"``.

``params`` has the following keys and values:

+-------+---------+-------------------+
| key   | type    | description       |
+=======+=========+===================+
| image | ``str`` | Path to image     |
+-------+---------+-------------------+
| text  | ``str`` | Text to vectorise |
+-------+---------+-------------------+

``training_step`` must return vector tensor.

``pre_finetuning`` takes no arguments.
