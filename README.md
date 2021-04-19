<h1 align="center">
   <a href="https://backprop.co">
      <img src=".github/header.png" width="300" alt="Backprop"/>
   </a>
</h1>

<p align="center">
   <a href="https://pypi.org/project/backprop/"><img src="https://img.shields.io/pypi/v/backprop"/></a> <img src="https://img.shields.io/pypi/pyversions/backprop"/> <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"/></a>
</p>

<p align="center">
Backprop makes it simple to use, finetune, and deploy state-of-the-art ML models.
</p>

<p align="center">
   <img src=".github/example.png" width="600"/>
</p>

Solve a variety of tasks with pre-trained models or finetune them in one line for your own tasks.

Out of the box tasks you can solve with Backprop:

- Conversational question answering in English
- Text Classification in 100+ languages
- Image Classification
- Text Vectorisation in 50+ languages
- Image Vectorisation
- Summarisation in English
- Emotion detection in English
- Text Generation

For more specific use cases, you can adapt a task with little data and a single line of code via finetuning.

| ⚡ [Getting started](#getting-started)                | Installation, few minute introduction                      |
| :---------------------------------------------------- | :-------------------------------------------------------- |
| 💡 [Examples](#examples)                              | Finetuning and usage examples                              |
| 📙 [Docs](https://backprop.readthedocs.io/en/latest/) | In-depth documentation about task inference and finetuning |
| ⚙️ [Models](https://backprop.co/hub)                   | Overview of available models                              |

## Getting started

### Installation

Install Backprop via PyPi:

```bash
pip install backprop
```

### Basic task inference

Tasks act as interfaces that let you easily use a variety of supported models.

```python
import backprop

context = "Take a look at the examples folder to see use cases!"

qa = backprop.QA()

# Start building!
answer = qa("Where can I see what to build?", context)

print(answer)
# Prints
"the examples folder"
```

You can run all tasks and models on your own machine, or in production with our inference [API](https://backprop.co), simply by specifying your `api_key`.

See how to use [all available tasks](https://backprop.readthedocs.io/en/latest/Tasks.html).

### Basic finetuning and uploading

Each task implements finetuning that lets you adapt a model for your specific use case in a single line of code.

A finetuned model is easy to upload to production, letting you focus on building great applications.

```python
from backprop.models import T5
from backprop import TextGeneration

tg = TextGeneration(T5)

# Any text works as training data
inp = ["I really liked the service I received!", "Meh, it was not impressive."]
out = ["positive", "negative"]

# Finetune with a single line of code
tg.finetune({"input_text": inp, "output_text": out})

# Use your trained model
prediction = tg("I enjoyed it!")

print(prediction)
# Prints
"positive"

# Upload to Backprop for production ready inference
# Describe your model
name = "t5-sentiment"
description = "Predicts positive and negative sentiment"

tg.upload(name=name, description=description, api_key="abc")
```

See [finetuning for other tasks](https://backprop.readthedocs.io/en/latest/Finetuning.html).

## Why Backprop?

1. No experience needed

   - Entrance to practical AI should be simple
   - Get state-of-the-art performance in your task without being an expert

2. Data is a bottleneck

   - Solve real world tasks without any data
   - With transfer learning, even a small amount of data can adapt a task to your niche requirements

3. There are an overwhelming amount of models

   - We offer a curated selection of the best open-source models and make them simple to use
   - A few general models can accomplish more with less optimisation

4. Deploying models cost effectively is hard work
   - If our models suit your use case, no deployment is needed: just call our API
   - Adapt and deploy your own model with just a few lines of code
   - Our API scales, is always available, and you only pay for usage

## Examples

- Solve any text based task with Finetuning ([Github](https://github.com/backprop-ai/backprop/blob/main/examples/Finetuning_GettingStarted.ipynb), [Colab](https://colab.research.google.com/github/backprop-ai/backprop/blob/main/examples/Finetuning_GettingStarted.ipynb))
- Search for images using text ([Github](https://github.com/backprop-ai/backprop/blob/main/examples/ImageVectorisation.ipynb))
- Finding answers from text ([Github](https://github.com/backprop-ai/backprop/blob/main/examples/Q%26A.ipynb))
- [More finetuning and task examples](https://github.com/backprop-ai/backprop/tree/main/examples)

## Documentation

Check out our [docs](https://backprop.readthedocs.io/en/latest/) for in-depth task inference and finetuning.

## Model Hub

Curated list of [state-of-the-art models](https://backprop.co/hub).

## Demos

Zero-shot image classification with [CLIP](https://clip.backprop.co).

## Credits

Backprop relies on many great libraries to work, most notably:

* [PyTorch](https://github.com/pytorch/pytorch)
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [Transformers](https://github.com/huggingface/transformers)
* [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
* [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
* [CLIP](https://github.com/openai/CLIP)

## Feedback

Found a bug or have ideas for new tasks and models? Open an [issue](https://github.com/backprop-ai/backprop/issues).
