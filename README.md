# Kiri Natural Language Engine

![PyPI](https://img.shields.io/pypi/v/kiri) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kiri) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Kiri Natural Language Engine is a high level Python library that makes using state-of-the-art language models easy, accessible and scalable.

With Kiri, no experience in AI is needed to solve valuable real world problems using:

- Semantic search (for ecommerce, documentation, etc.)
- Conversational question answering (for FAQ chatbots, text analysis, etc.)
- Zero-shot classification (for email sorting, intent detection, etc.)
- Summarisation (TLDRs for long documents)
- Emotion detection (for customer satisfaction, text analysis, etc.)

Run everything locally or take your code to production using our free, optimised inference [API](https://kiri.ai).

| ⚡ [Getting started](#getting-started) | Installation, few minute introduction     |
| -------------------------------------- | :---------------------------------------- |
| 💡 [Examples](#examples)               | Sample problems solved using Kiri         |
| 📙 [Docs](#documentation)              | In-depth documentation for advanced usage |

## Getting started

### Installation

Install Kiri via PyPi:

```bash
pip install kiri
```

### Basic usage

```python
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

```

## Examples

Take a look at the [examples folder](/examples).

## Documentation

Check out our [docs]().
