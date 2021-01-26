Kiri Core: Documents
====================

Documents are the base object on which much of Kiri operates.

At their simplest, they're a wrapper for a string of content and an associated content vector.

Content Vectors
---------------
Upon uploading a document to a `DocStore`, it is processed -- the `vectorise_model` parameter of the Kiri Core is used 
to encode the entire document as a vector, which can then used for semantic search.

Document Attributes
-------------------
Document attributes can be any metadata that you might require -- title, publication date, url, etc.

They won't be vectorised, and are accessible upon retrieval from a `DocStore`.

ChunkedDocuments
----------------
The basic `Document` has a content vector field. However, for longer documents or specific tasks, 
more granular vectorisation is beneficial.

`ChunkedDocuments` accept a `chunking_level`. This parameter determines how many sentences make up a single chunk.
The content text is split into chunks of sentences, and each chunk is vectorised individually.

ElasticDocument Variants
------------------------
Both the `Document` and `ChunkedDocument` classes have Elastic-ready variants. These variants are used in much the same way as their 
standard counterparts: however, they have in-built functionality for creating Elastic mappings, as well as storage and retrieval to/from an Elastic index backend.

They were designed to make it as painless as possible to transition a demo with standard `Documents` to a production-ready backend.