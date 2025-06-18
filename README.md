# RNN-Imdb-Sentiment-Analysis-
This project predicts the sentiment (positive or negative) of movie reviews using a deep learning model trained on the IMDB dataset. It preprocesses text, uses an RNN for classification, and provides a Streamlit web app for real-time predictions, making sentiment analysis easy and accessible.
# IMDB Movie Review Sentiment Analysis

This project predicts the sentiment (positive or negative) of movie reviews using a deep learning model trained on the IMDB dataset. It features a user-friendly web interface built with Streamlit for real-time sentiment analysis.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Demo](#demo)
- [License](#license)

---

## Overview

This project uses a Recurrent Neural Network (RNN) to classify movie reviews from the IMDB dataset as either positive or negative. The model is trained on thousands of labeled reviews and deployed as a web app using Streamlit, allowing users to input their own reviews and instantly receive sentiment predictions.

---

## Features

- Preprocessing of text data (tokenization, padding)
- RNN-based sentiment classification
- Model training and evaluation
- Streamlit web app for interactive predictions
- Confidence score for each prediction

---

## Dataset

The [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/) contains 50,000 movie reviews labeled as positive or negative. The dataset is preprocessed so that each word is mapped to a unique integer, and reviews are padded to a fixed length for model input.

---

## How It Works

1. **Preprocessing:**  
   - Convert words to integer indices using a word index.
   - Pad or truncate reviews to a fixed length (e.g., 500 words).

2. **Model Training:**  
   - Use an Embedding layer to represent words as dense vectors.
   - Train an RNN (such as LSTM or SimpleRNN) on the training data.
   - Evaluate the model on validation data.

3. **Prediction:**  
   - User inputs a review in the Streamlit app.
   - The review is preprocessed and fed to the trained model.
   - The app displays the predicted sentiment and confidence score.

---

## Model Architecture

- **Embedding Layer:** Converts word indices to dense vectors.
- **RNN Layer:** Processes the sequence of word vectors.
- **Dense Output Layer:** Outputs a probability for positive sentiment.

Example:
```python
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(SimpleRNN(64))
model.add(Dense(1, activation='sigmoid'))
```

---

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)

Run the training notebook or script to train and save the model:
```bash
python train_model.py
```
Or use the provided pre-trained model (`imdb_rnn_model.h5`).

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

### 5. Use the Web App

- Enter a movie review in the text box.
- Click "Predict Sentiment" to see the result.

---

## Project Structure

```
├── app.py
├── imdb_rnn_model.h5
