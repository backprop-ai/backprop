Utils
=====

Functions from ``backprop.utils`` let you save, load and upload models.

Before saving or uploading a model be sure to have added the model a name, description and list of tasks.

The name must be between 3 and 100 lowercase a-z characters, with numbers, dashes (-) and underscores (\_) allowed. 

The description can be any string.

See the available task strings in :ref:`models`.

.. code-block:: python

    model.name = "some-string"
    model.description = "Some description about what the model can do"
    model.tasks = ["text-classification"]

See the full reference in :ref:`backprop-utils`.

Save
----

.. automodule:: backprop.utils.save
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. code-block:: python

    import backprop

    # Saves model instance to ~/.cache/backprop/model_name
    # model_name is determined from model_instance.name
    backprop.save(model_instance)

Load
----

.. automodule:: backprop.utils.load
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. code-block:: python

    import backprop

    # Loads model instance from ~/.cache/backprop/model_name
    model = backprop.load("model_name")


Upload
------

For a successful upload, ensure that the model is valid by following the check list in :ref:`models`.

.. automodule:: backprop.utils.upload
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. code-block:: python

    import backprop

    # Uploads model to Backprop for production ready inference
    backprop.upload(model, api_key="abc")