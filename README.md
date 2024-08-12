# English To Telugu Machine Translation Using Transformers

This project demonstrates how to build a machine translation model from English to Telugu using the Transformer architecture. The project covers everything from designing the Transformer model from scratch, preprocessing custom data, training the model, and finally using the model for English to Telugu translation.

* Table of Contents
* Project Overview
* Architecture
* Data Preprocessing
* Training
* Translation
* Installation
* License

  
## Project Overview
Machine Translation (MT) is a sub-field of Natural Language Processing (NLP) that focuses on automatically translating text from one language to another. This project uses the Transformer model, an attention-based architecture, to perform translation from English to Telugu.

## Architecture
The Transformer model was introduced in the paper "Attention Is All You Need" by Vaswani et al. Unlike previous sequence-to-sequence models, Transformers rely entirely on self-attention mechanisms to capture the relationships between words in a sentence, making them highly effective for tasks like machine translation.

## Key Components
Positional Encoding: Since Transformers do not inherently understand the order of sequences, positional encodings are added to the input embeddings.
Multi-Head Attention: This allows the model to focus on different parts of the input sequence simultaneously.
Feed-Forward Networks: Dense layers applied independently to each position in the sequence.
Encoder-Decoder Structure: The encoder processes the input (English), and the decoder generates the output (Telugu).'

## Data Preprocessing:
Before training the model, the data must be preprocessed to convert raw text into a format suitable for the Transformer model.

Data Collection: Collect parallel English-Telugu text data. Ensure that the data is cleaned and aligned correctly.
Tokenization: Use SpaCy or a custom tokenizer to split sentences into tokens (words or subwords).
Vocabulary Creation: Build vocabularies for both English and Telugu using a function that considers the frequency of words.
Padding and Batching: Pad sequences to ensure they are of uniform length and create batches for efficient training.

## Training
The training process involves feeding the preprocessed data into the Transformer model, computing the loss, and updating the model's parameters using backpropagation.

Model Initialization: Initialize the Transformer model with the appropriate dimensions, layers, and hyperparameters.
Loss Function: Use a suitable loss function, such as Cross-Entropy Loss, to measure the difference between the predicted and actual translations.
Optimizer: Use Adam or a similar optimizer for updating the model parameters.
Training Loop: Iterate through the dataset for multiple epochs, performing forward and backward passes, and updating the model parameters.
Checkpointing: Save the model's state periodically to avoid losing progress.

## Translation
After training, the model can be used for translating English sentences to Telugu.

Input Sentence: Preprocess the input English sentence.
Generate Translation: Pass the sentence through the Transformer model to generate the Telugu translation.
Post-Processing: Convert the output tokens back into human-readable text.

## Installation
To set up the project locally, follow these steps:

Clone the repository:
