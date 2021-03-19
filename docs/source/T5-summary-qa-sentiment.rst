T5 for Summary, Q&A, and Sentiment
==================================

Backprop is built to be flexible -- while we've included some useful pre-trained models for tasks, we want users to be able to 
easily deploy their own models, built to solve their own problems.

Here, we include some info on the training of our T5 model for summary, Q&A, and sentiment analysis.

Model Info
----------------------------------

**Base model/tokenizer**: `T5 Base <https://huggingface.co/t5-base>`_

**Training data**:

- `CoQa <https://stanfordnlp.github.io/coqa/>`_
- `SQuAD2.0 <https://rajpurkar.github.io/SQuAD-explorer/>`_
- `GoEmotions <https://github.com/google-research/google-research/tree/master/goemotions>`_
- `CNN/DailyMail <https://github.com/abisee/cnn-dailymail>`_


With this training data, the model is able to achieve a score of **F1 79.5** on SQuAD2.0, and **F1 70.6** on CoQa.

Evaluations on summarisation and sentiment analysis have not yet been performed.

The learning rate while training was ``0.001``.

Other parameters, such as epochs and batch size, were variable across training and will require experimentation for exact replication.