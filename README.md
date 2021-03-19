<h1 align="center">Backprop</h1>

<p align="center">
   <a href="https://pypi.org/project/backprop/"><img src="https://img.shields.io/pypi/v/backprop"/></a> <img src="https://img.shields.io/pypi/pyversions/backprop"/> <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"/></a>
</p>

<p align="center">
Backprop is a Python library that makes it simple to solve AI tasks with state-of-the-art machine learning models.
</p>

Backprop is built around solving tasks with transfer learning. It implements advanced models that are general enough to solve real world tasks with minimal data required from the user.

<p align="center">
   <img src=".github/example.png" width="700"/>
</p>

Out of the box tasks you can solve with Backprop:

- Conversational question answering in English (for FAQ chatbots, text analysis, etc.)
- Text Classification in 100+ languages (for email sorting, intent detection, etc.)
- Image Classification (for object recognition, OCR, etc.)
- Text Vectorisation in 50+ languages (semantic search for ecommerce, documentation, etc.)
- Summarisation in English (TLDRs for long documents)
- Emotion detection in English (for customer satisfaction, text analysis, etc.)
- Text Generation (for idea, story generation and broad task solving)

For more specific use cases, you can adapt a task with little data and a few lines of code via finetuning. We are working to add finetuning to all our available tasks.

You can run all tasks on your own machine, or in production with our optimised inference [API](https://backprop.co), where you only pay for usage. It includes all the tasks & models in our library, and allows you to upload your own finetuned models.

| ⚡ [Getting started](#getting-started)                                    | Installation, few minute introduction     |
| :------------------------------------------------------------------------ | :---------------------------------------- |
| 💡 [Examples](https://github.com/backprop-ai/backprop/tree/main/examples) | Sample problems solved using Backprop     |
| 📙 [Docs](https://backprop.readthedocs.io/en/latest/)                     | In-depth documentation for advanced usage |

## Getting started

### Installation

Install Backprop via PyPi:

```bash
pip install backprop
```

### Basic task solving

```python
from backprop import QA

context = "Take a look at the examples folder to see use cases!"

qa = QA()

# Start building!
answer = qa("Where can I see what to build?", context)

print(answer)
# Prints
"the examples folder"
```

### Basic finetuning and uploading

```python
from backprop.models import T5
from backprop import TextGeneration

tg = TextGeneration(T5)

# Any text works as training data
inp = ["I really liked the service I received!", "Meh, it was not impressive."]
out = ["positive", "negative"]

# Finetune with a single line of code
tg.finetune(inp, out)

# Use your trained model
prediction = tg("I enjoyed it!")

print(prediction)
# Prints
"positive"

# Upload to Backprop for production ready inference

model = tg.model
# Describe your model
model.name = "t5-sentiment"
model.description = "Predicts positive and negative sentiment"

backprop.upload(model, api_key="abc")
```

## Why Backprop?

1. No experience needed

   - Entrance to practical AI should be simple
   - Get state-of-the-art performance in your task without being an expert

2. Data is a bottleneck

   - Use AI without needing access to "big data"
   - With transfer learning, even a small amount of data can adapt a task to your niche requirements

3. There are an overwhelming amount of models

   - We implement the best ones and make them simple to use
   - A few general models can accomplish more with less optimisation

4. Deploying models cost effectively is hard work
   - If our models suit your use case, no deployment is needed: just call our API
   - Adapt and deploy your own model with just a few lines of code
   - Our API scales, is always available, and you only pay for usage

## Examples

Take a look at the [examples folder](https://github.com/backprop-ai/backprop/tree/main/examples).

## Documentation

Check out our [docs](https://backprop.readthedocs.io/en/latest/).
