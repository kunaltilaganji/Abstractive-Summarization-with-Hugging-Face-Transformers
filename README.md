# Pretraining-BERT-with-Hugging-Face-Transformers

# Introduction

Automatic summarization is a central problem in Natural Language Processing (NLP). It involves challenges related to language understanding and generation. This tutorial focuses on abstractive summarization, aiming to generate concise, abstractive summaries of news articles.

We tackle this task using the [Text-to-Text Transfer Transformer (T5)](https://arxiv.org/abs/1910.10683), a Transformer-based model pretrained on various text-to-text tasks. T5's encoder-decoder architecture has shown impressive results in sequence-to-sequence tasks like summarization and translation.

In this notebook, I fine-tune pretrained T5 on the Abstractive Summarization task using Hugging Face Transformers and the `XSum` dataset.

## Setup

Install the required libraries:

```bash
pip install transformers==4.20.0
pip install keras_nlp==0.3.0
pip install datasets
pip install huggingface-hub
pip install nltk
pip install rouge-score
```

## Loading the Dataset

We download the [Extreme Summarization (XSum)](https://arxiv.org/abs/1808.08745) dataset, consisting of BBC articles and single-sentence summaries. The dataset is divided into training, validation, and test sets. We use the ROUGE metric for evaluation.

We use the [Hugging Face Datasets](https://github.com/huggingface/datasets) library to easily load the data with `load_dataset`.

## Data Pre-processing

Before feeding texts to the model, we pre-process them using Hugging Face Transformers' `Tokenizer`. It tokenizes inputs, converts tokens to IDs in the pretrained vocabulary, and generates model-ready inputs.

```python
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT)
```

## Inference

We use the `summarization` pipeline from Hugging Face Transformers to infer the trained model's summary for arbitrary articles.

```python
from transformers import pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="tf")
summary = summarizer(article)
```
Feel free to use this README as a template for your GitHub repository and customize it with additional details, instructions, and explanations.

<b>Please replace the placeholders like `MODEL_CHECKPOINT` with actual values and adjust any other details to match your repository's content.</b>


## Acknowledgments
This project was inspired by the research papers  [Text-to-Text Transfer Transformer (T5)](https://arxiv.org/abs/1910.10683).


## Contact

For questions or feedback, please feel free to reach out:

- Author: Kunal Tilaganji 

Project Link: [Link](https://github.com/kunaltilaganji/Abstractive-Summarization-with-Hugging-Face-Transformers)