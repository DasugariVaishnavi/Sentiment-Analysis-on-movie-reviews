# ============================================================
# MOVIE REVIEW SENTIMENT ANALYSIS
# Using NLP, TF-IDF, Logistic Regression and Linear SVM
# Suitable for VS Code
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import os
import re
import string
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from nltk.corpus import stopwords


# ------------------------------------------------------------
# 2. DOWNLOAD NLTK STOPWORDS
# ------------------------------------------------------------

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

RANDOM_STATE = 42


# ------------------------------------------------------------
# 3. DATASET PATH
# ------------------------------------------------------------

# Keep "IMDB Dataset.csv" in the same folder as this Python file.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "IMDB Dataset.csv")

print("Dataset path:")
print(CSV_PATH)


# ------------------------------------------------------------
# 4. CHECK WHETHER DATASET EXISTS
# ------------------------------------------------------------

if not os.path.exists(CSV_PATH):
    print("\nERROR: IMDB Dataset.csv was not found.")
    print("Please place 'IMDB Dataset.csv' in the same folder as sentiment.py")
    exit()


# ------------------------------------------------------------
# 5. LOAD DATASET
# ------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(CSV_PATH)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 6. STANDARDIZE COLUMN NAMES
# ------------------------------------------------------------

def standardize_columns(df):

    text_columns = [
        c for c in df.columns
        if c.lower() in ("review", "text", "content", "sentence")
    ]

    label_columns = [
        c for c in df.columns
        if c.lower() in (
            "sentiment",
            "label",
            "target",
            "polarity",
            "class"
        )
    ]

    if text_columns and label_columns:

        df = df.rename(
            columns={
                text_columns[0]: "review",
                label_columns[0]: "sentiment"
            }
        )

    return df


df = standardize_columns(df)


# ------------------------------------------------------------
# 7. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

if "review" not in df.columns or "sentiment" not in df.columns:

    print("\nERROR:")
    print("The dataset must contain 'review' and 'sentiment' columns.")

    print("\nAvailable columns:")
    print(df.columns.tolist())

    exit()


# ------------------------------------------------------------
# 8. REMOVE MISSING AND DUPLICATE DATA
# ------------------------------------------------------------

before = len(df)

df = (
    df
    .dropna(subset=["review", "sentiment"])
    .drop_duplicates(subset=["review"])
    .reset_index(drop=True)
)

after = len(df)

print(f"\nRows kept: {after}/{before}")


# ------------------------------------------------------------
# 9. ENCODE SENTIMENT LABELS
# ------------------------------------------------------------

if df["sentiment"].dtype == "object":

    df["sentiment"] = (
        df["sentiment"]
        .astype(str)
        .str.lower()
        .map({
            "positive": 1,
            "pos": 1,
            "negative": 0,
            "neg": 0
        })
    )


df["sentiment"] = pd.to_numeric(
    df["sentiment"],
    errors="coerce"
)

df = df.dropna(subset=["sentiment"])

df["sentiment"] = df["sentiment"].astype(int)


print("\nSentiment distribution:")
print(df["sentiment"].value_counts())


# ------------------------------------------------------------
# 10. TEXT PREPROCESSING FUNCTION
# ------------------------------------------------------------

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove stopwords
    words = [
        word
        for word in text.strip().split()
        if word not in stop_words
    ]

    return " ".join(words)


# ------------------------------------------------------------
# 11. APPLY TEXT PREPROCESSING
# ------------------------------------------------------------

print("\nCleaning reviews...")

df["clean_review"] = df["review"].astype(str).apply(clean_text)


print("\nOriginal and cleaned reviews:")

print(
    df[
        ["review", "clean_review", "sentiment"]
    ].head()
)


# ------------------------------------------------------------
# 12. PREPARE FEATURES AND LABELS
# ------------------------------------------------------------

X = df["clean_review"].values

y = df["sentiment"].values


# ------------------------------------------------------------
# 13. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# 14. TF-IDF SETTINGS
# ------------------------------------------------------------

tfidf_args = {

    "max_features": 20000,

    # Unigrams + bigrams
    "ngram_range": (1, 2),

    # Ignore very rare words
    "min_df": 2
}


# ------------------------------------------------------------
# 15. LOGISTIC REGRESSION PIPELINE
# ------------------------------------------------------------

pipe_lr = Pipeline([

    (
        "tfidf",
        TfidfVectorizer(**tfidf_args)
    ),

    (
        "clf",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    )
])


# ------------------------------------------------------------
# 16. LINEAR SVM PIPELINE
# ------------------------------------------------------------

pipe_svm = Pipeline([

    (
        "tfidf",
        TfidfVectorizer(**tfidf_args)
    ),

    (
        "clf",
        LinearSVC(
            random_state=RANDOM_STATE
        )
    )
])


# ------------------------------------------------------------
# 17. TRAIN LOGISTIC REGRESSION
# ------------------------------------------------------------

print("\nTraining Logistic Regression...")

pipe_lr.fit(
    X_train,
    y_train
)

print("Logistic Regression training completed.")


# ------------------------------------------------------------
# 18. TRAIN LINEAR SVM
# ------------------------------------------------------------

print("\nTraining Linear SVM...")

pipe_svm.fit(
    X_train,
    y_train
)

print("Linear SVM training completed.")


# ------------------------------------------------------------
# 19. MODEL EVALUATION FUNCTION
# ------------------------------------------------------------

def evaluate(model, X_test, y_test, title):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    print("\n" + "=" * 60)

    print(title)

    print("=" * 60)

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"F1-score: {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            digits=4
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Negative",
            "Positive"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    disp.plot(
        ax=ax,
        values_format="d"
    )

    plt.title(
        f"Confusion Matrix - {title}"
    )

    plt.tight_layout()

    plt.show()

    return accuracy, f1


# ------------------------------------------------------------
# 20. EVALUATE LOGISTIC REGRESSION
# ------------------------------------------------------------

acc_lr, f1_lr = evaluate(
    pipe_lr,
    X_test,
    y_test,
    "Logistic Regression"
)


# ------------------------------------------------------------
# 21. EVALUATE LINEAR SVM
# ------------------------------------------------------------

acc_svm, f1_svm = evaluate(
    pipe_svm,
    X_test,
    y_test,
    "Linear SVM"
)


# ------------------------------------------------------------
# 22. 5-FOLD CROSS VALIDATION
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)


for name, model in [
    ("Logistic Regression", pipe_lr),
    ("Linear SVM", pipe_svm)
]:

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="f1"
    )

    print(
        f"\n{name}"
    )

    print(
        f"F1 scores: {scores}"
    )

    print(
        f"Mean F1: {scores.mean():.4f}"
    )

    print(
        f"Standard Deviation: {scores.std():.4f}"
    )


# ------------------------------------------------------------
# 23. SELECT BEST MODEL
# ------------------------------------------------------------

if f1_svm >= f1_lr:

    best_model = pipe_svm

    best_name = "linear_svm"

    print(
        "\nBest model: Linear SVM"
    )

else:

    best_model = pipe_lr

    best_name = "logistic_regression"

    print(
        "\nBest model: Logistic Regression"
    )


# ------------------------------------------------------------
# 24. SAVE BEST MODEL
# ------------------------------------------------------------

MODEL_PATH = os.path.join(
    BASE_DIR,
    f"sentiment_{best_name}_pipeline.joblib"
)

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nModel saved successfully!")

print(
    "Model location:"
)

print(
    MODEL_PATH
)


# ------------------------------------------------------------
# 25. LOAD SAVED MODEL
# ------------------------------------------------------------

loaded_model = joblib.load(
    MODEL_PATH
)


# ------------------------------------------------------------
# 26. SENTIMENT PREDICTION FUNCTION
# ------------------------------------------------------------

def predict_sentiment(texts):

    if isinstance(texts, str):

        texts = [texts]

    cleaned_texts = [
        clean_text(text)
        for text in texts
    ]

    predictions = loaded_model.predict(
        cleaned_texts
    )

    results = []

    for prediction in predictions:

        if prediction == 1:

            results.append("Positive")

        else:

            results.append("Negative")

    return results


# ------------------------------------------------------------
# 27. TEST WITH SAMPLE REVIEWS
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)


sample_reviews = [

    "This movie was amazing and I really enjoyed it.",

    "The movie was boring and disappointing.",

    "Excellent acting and a fantastic story.",

    "I did not like this movie at all."
]


sample_results = predict_sentiment(
    sample_reviews
)


for review, result in zip(
    sample_reviews,
    sample_results
):

    print("\nReview:")
    print(review)

    print("Prediction:")
    print(result)


# ------------------------------------------------------------
# 28. COMMAND-LINE USER INPUT
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("MOVIE REVIEW SENTIMENT PREDICTOR")
print("=" * 60)

user_text = input(
    "\nEnter a movie review: "
)

result = predict_sentiment(
    user_text
)[0]

print(
    f"\nPredicted Sentiment: {result}"
)


# ------------------------------------------------------------
# 29. GRADIO WEB APPLICATION
# ------------------------------------------------------------

print("\nStarting Gradio application...")


import gradio as gr


def gradio_predict(review):

    if not review or not review.strip():

        return "Please enter a movie review."

    return predict_sentiment(
        review
    )[0]


demo = gr.Interface(

    fn=gradio_predict,

    inputs=gr.Textbox(
        lines=5,
        placeholder="Type a movie review here..."
    ),

    outputs=gr.Textbox(
        label="Predicted Sentiment"
    ),

    title="🎬 Movie Review Sentiment Analysis",

    description=(
        "Enter a movie review and the "
        "machine learning model will predict "
        "whether the sentiment is Positive or Negative."
    )
)


# ------------------------------------------------------------
# 30. LAUNCH GRADIO
# ------------------------------------------------------------

demo.launch()