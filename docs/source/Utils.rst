Utils
=====

Functions from ``kiri.utils`` let you save, load and upload models.

Before saving or uploading a model be sure to have added the model a name, description and list of tasks.

The name must be between 3 and 100 lowercase a-z characters, with numbers, dashes (-) and underscores (\_) allowed. 

The description can be any string.

See the available task strings in :ref:`models`.

.. code-block:: python

    model.name = "some-string"
    model.description = "Some description about what the model can do"
    model.tasks = ["text-classification"]

See the full reference in :ref:`kiri-utils`.

Save
----

.. automodule:: kiri.utils.save
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. code-block:: python

    import kiri

    # Saves model instance to ~/.cache/kiri/model_name
    # model_name is determined from model_instance.name
    kiri.save(model_instance)

Load
----

.. automodule:: kiri.utils.load
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. code-block:: python

    import kiri

    # Loads model instance from ~/.cache/kiri/model_name
    model = kiri.load("model_name")


Upload
------

For a successful upload, ensure that the model is valid by following the check list in :ref:`models`.

.. automodule:: kiri.utils.upload
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. code-block:: python

    import kiri

    # Uploads model to Kiri for production ready inference
    kiri.upload(model, api_key="abc")