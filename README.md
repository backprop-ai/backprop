<h1 align="center">Kiri</h1>

<p align="center">
   <a href="https://pypi.org/project/kiri/"><img src="https://img.shields.io/pypi/v/kiri"/></a> <img src="https://img.shields.io/pypi/pyversions/kiri"/> <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"/></a>
</p>

<p align="center">
Kiri is a Python library that makes using state-of-the-art AI easy, accessible and scalable.
</p>

<p align="center">
   <img src=".github/kiri-example.png" width="600"/>
</p>

With Kiri, no experience in AI is needed to solve valuable real world problems using:

- Conversational question answering in English (for FAQ chatbots, text analysis, etc.)
- Zero-shot text classification in 100+ languages (for email sorting, intent detection, etc.)
- Zero-shot image classification (for object recognition, OCR, etc.)
- Text vectorisation in 50+ languages (semantic search for ecommerce, documentation, etc.)
- Summarisation in English (TLDRs for long documents)
- Emotion detection in English (for customer satisfaction, text analysis, etc.)

Run everything locally or take your code to production using our optimised inference [API](https://kiri.ai), where you only pay for usage.

| ⚡ [Getting started](#getting-started)                            | Installation, few minute introduction     |
| :---------------------------------------------------------------- | :---------------------------------------- |
| 💡 [Examples](https://github.com/kiri-ai/kiri/tree/main/examples) | Sample problems solved using Kiri         |
| 📙 [Docs](https://kiri.readthedocs.io/en/latest/)                 | In-depth documentation for advanced usage |

## Getting started

### Installation

Install Kiri via PyPi:

```bash
pip install kiri
```

### Basic usage

```python
from kiri import Kiri

context = "Take a look at the examples folder to see use cases!"

# Use our inference API
kiri = Kiri(api_key="abc")
# Or run locally
kiri = Kiri(local=True)

# Start building!
answer = kiri.qa("Where can I see what to build?", context)

print(answer)
# Prints
"the examples folder"
```

## Why Kiri?

1. No experience needed

   - Entrance to practical AI should be simple
   - Get state-of-the-art performance in your task without being an expert

2. There is an overwhelming amount of models

   - We implement the best ones for various tasks
   - A few general models can accomplish more with less optimisation

3. Deploying models cost effectively is hard work
   - If our models suit your use case, no deployment is needed
   - Our API scales, is always available, and you only pay for usage

## Examples

Take a look at the [examples folder](https://github.com/kiri-ai/kiri/tree/main/examples).

## Documentation

Check out our [docs](https://kiri.readthedocs.io/en/latest/).
