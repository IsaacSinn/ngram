# Goal
1. Ngram model with absolute discounting with recursive backoff from scratch
2. RNN model with pytorch
3. LSTM model with pytorch

# Project Instructions

## Overview
This project implements three modeling approaches:
- **N-gram Model**: Uses an absolute discounting with backoff algorithm to handle unseen n-grams.
- **RNN Model**: Implements a basic RNN with a hidden layer of size 64 (other parameters are kept at their defaults) and a linear layer mapping to the vocabulary size. Trained using the cross-entropy loss with the Adam optimizer (lr=1e-3).
- **LSTM Model**: Implements an LSTM using an LSTMCell with hidden size 64, along with a corresponding linear output layer and similar optimization settings as the RNN.

## How to Train and Run

### N-gram Model
1. Preprocess your text data.
2. Build n-gram frequency tables.
3. Apply absolute discounting with backoff to compute probabilities.
4. Run the training script by executing:
   ```
   python prompt/experiment_ngram.py
   ```
   This calls the appropriate main function to train and test the n-gram model.

### RNN Model
1. Prepare training data in the expected format.
2. Run the training script (e.g., from `prompt/experiment_rnn.py`) to train the RNN.
3. The RNN uses an RNNCell with hidden size 64 and a fully connected layer mapping to the vocabulary.
4. Model checkpoints are saved after each epoch.

### LSTM Model
1. Similar to the RNN, format your training data accordingly.
2. Use the training script (from `prompt/experiment_rnn.py`) to train the LSTM model.
3. The LSTM uses an LSTMCell with hidden size 64 and a corresponding linear layer.
4. Training and evaluation include accuracy calculations on a test set.

## Running the Models
- Ensure that the necessary packages (e.g., PyTorch) and data files are available.
- Use the command:
  ```
  python prompt/experiment_rnn.py
  ```
  to train and test the RNN or LSTM model.
- For the n-gram model, run the respective script:
  ```
  python prompt/experiment_ngram.py
  ```
  after verifying its setup.

Happy coding!
