Quickstart
==========

Installation
------------

Install Kiri via PyPi

.. code-block ::

    pip install kiri

Basic Usage
-----------

.. code-block:: python

    from kiri import Kiri, Document

    # Unprocessed documents
    documents = [
        Document("Look at examples to see awesome use cases!"),
        Document("Check out the docs to see what's possible!")
    ]

    # Use our inference API
    kiri = Kiri(api_key="abc")
    # Or run locally
    kiri = Kiri(local=True)

    # Process documents
    kiri.upload(documents)

    # Start building!
    search_results = kiri.search("What are some cool apps that have been built?")

    print(search_results.to_json())

    # Prints
    {
    "max_score": 0.3804888461635889,
    "total_results": 2,
    "results": [
        {
            "document": {
                "id":"LzhtWcpV2eoMk8GJwaw7na",
                "content":"Look at examples to see awesome use cases!"
            },
            "score": 0.3804888461635889,
            "preview":" Look at examples to see awesome use cases!"
        },
        {
            "document": {
                "id":"bcLb8xUK585Zm6rZrwj89A",
                "content":"Check out the docs to see what's possible!"
            },
            "score": 0.1742559312454076,
            "preview":" Check out the docs to see what's possible!"
        }
      ]
    }

Why Kiri?
---------

1. No Experience needed
    
    - Entrance to practical AI should be simple
    - Get state-of-the-art performance in your task, without being an expert

2. There are an overwheleming amount of models
    
    - We implement the best ones for various use cases
    - A few general models can accomplish more with less optimisation

3. Deploying models cost-effectively is hard work

    - If our models suit your use case, no deployment is needed
    - Our API scales, is always available, and you only pay for usage