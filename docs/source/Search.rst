Kiri Core: Semantic Search
==========================
What's semantic search?
-----------------------
In a 'typical' search system, search is performed by matching keywords. While sometimes useful, this often requires guesswork on behalf of the person searching.
They might not be aware of what content exists in the collection they're searching, and have to hope that their queries match words or phrases in the documents.

Semantic search, alternatively, uses word embeddings to understand what your query *means*, and compare that to embeddings of documents. This allows the search engine
to find results that closely match the meaning of the query, regardless of exact phrasing and words used. 


Included in Kiri
~~~~~~~~~~~~~~~~
We've got two flavors of search bundled in: In-memory, and Elastic-based search.

The in-memory version is meant for local testing, experimenting -- dev stuff.

Using an Elastic backend is more suitable for production.


What's Elastic?
---------------
Elasticsearch is an open source search engine -- it's commonly used in enterprise.
It's quick, scalable, and documents are schema-free. 

Elastic uses the BM25 bag-of-words model in combination with our semantic search to achieve the best performance possible.


Why use Elastic with Kiri?
---------------------
Elastic provides a production-ready backend for search with very little overhead required.

When used in combination with Kiri, you get all the benefits of Elastic, enhanced with Kiri's
semantic processing.


Setting up Elastic
------------------
With Docker
~~~~~~~~~~~
There's a simple Docker one-liner that can be used to instantly set up Elasticsearch.

`docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.3`

The `-d` flag refers to detached mode, meaning it will run in the background. Omit `-d` to get logs in your terminal instance.

If you'd like to host the Docker instance remotely, AWS has a small free-tier instance. It has 750 monthly hours, so you can leave it constantly running.

Local Only
~~~~~~~~~~
Refer to this_ guide to get Elasticsearch running locally on your computer.

.. _this: https://www.elastic.co/guide/en/elasticsearch/reference/current/getting-started-install.html

Example With Code
-----------------
Refer to our repo examples_ to see how search works.

.. _examples: https://github.com/kiri-ai/kiri/blob/main/examples/core/Search.ipynb