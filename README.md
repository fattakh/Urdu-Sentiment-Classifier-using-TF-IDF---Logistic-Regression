Urdu Sentiment Analysis using TF-IDF & Logistic Regression

This project implements a simple and effective Urdu sentiment classification model using scikit-learn. It demonstrates how to build a text-processing pipeline, train a machine-learning classifier, evaluate performance, and generate predictions for new Urdu text inputs.

🚀 Project Overview

This repository contains:

A small labeled Urdu dataset (positive & negative sentiment)

A complete ML pipeline using:

TF-IDF Vectorization (1–2 n-grams)

Logistic Regression Classifier

Training/testing split with stratification

Evaluation using classification_report

Example predictions on unseen Urdu sentences

The goal is to provide a minimal, easy-to-understand sentiment analysis baseline for Urdu NLP tasks.

📂 Code Explanation

The pipeline includes:

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
    ("clf", LogisticRegression(max_iter=200))
])


This setup automatically transforms text into numerical TF-IDF features and feeds them into a logistic regression model for classification.

🧪 How It Works

Dataset Creation
Six Urdu sentences manually labeled as positive or negative.

Train/Test Split
60% training, 40% testing (stratified by label).

Model Training
The pipeline fits the TF-IDF vectorizer and trains the logistic model.

Model Evaluation
Outputs precision, recall, F1-score.

Prediction on Samples
The model predicts sentiment for two new Urdu sentences.

📊 Sample Prediction
samples = ["سروس بہترین ہے", "پیسوں کے ضیاع کے برابر"]
print(pipe.predict(samples))

📦 Requirements

Install the required Python libraries:

pip install pandas scikit-learn

▶️ Running the Code

Simply clone the project and run:

python main.py
