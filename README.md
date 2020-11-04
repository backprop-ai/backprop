# Kiri Natural Language Engine

![PyPI](https://img.shields.io/pypi/v/kiri) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kiri)[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 

## Getting started

### Installation

Install Kiri via PyPi:

```bash
pip install kiri
```

For more information about the models used in this package, check out our [organization](https://huggingface.co/kiri-ai) page on huggingface.



### Usage 

```python
from kiri import Kiri, ElasticDocStore, ElasticChunkedDocument

# Unvectorized documents
documents = [
    ElasticDocument(content="This is amazing content"), 	      
    ElasticDocument(content="This is some other interesting content")
]

# Pass the url to the elastic server.
elastic_url = "http://localhost:9200"
doc_store = ElasticDocStore(elastic_url, doc_store=ElasticChunkedDocument)

kiri = Kiri(doc_store)

elastic_docs = [ElasticChunkedDocument(doc["content"]) for doc in documents]

# Upload documents to store
kiri.upload(elastic_docs)


```



## Documentation

For more in-depth documentation on the package, check out our official [docs]() page made with Hugo.



