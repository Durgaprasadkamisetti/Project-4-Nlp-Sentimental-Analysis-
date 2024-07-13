
# -*- coding: utf-8 -*-
"""
Streamlit App for Sentiment Analysis
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS
import re

# Title and Description
st.title('Sentiment Analysis of Reviews')
st.write('This app performs sentiment analysis on a dataset of reviews and visualizes the results.')

# Text Input for Single Review
st.write('## Enter a Review for Sentiment Prediction')
user_review = st.text_area("Type your review here...")

# Pre-trained models
@st.cache(allow_output_mutation=True)
def load_models_and_vectorizer():
    # Load your dataset for training
    df = pd.read_csv('reviews.csv')

    # Data Cleaning
    df.fillna('', inplace=True)
    df['cleanedReviewBody'] = df['reviewBody'].apply(lambda x: ' '.join(
        word for word in x.lower().split() if word.isalpha()))

    # Merge columns for analysis
    df['mergedReview'] = df['headline'] + ". " + df['cleanedReviewBody']

    # Encode sentiment
    df['ratingValue'] = pd.to_numeric(df['ratingValue'], errors='coerce')
    df = df.dropna(subset=['ratingValue'])
    df['sentiment'] = df['ratingValue'].apply(
        lambda x: 'Negative' if x < 3 else ('Neutral' if x == 3 else 'Positive'))

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['mergedReview'])
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine": SVC(probability=True),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models, vectorizer, X_test, y_test

models, vectorizer, X_test, y_test = load_models_and_vectorizer()

def contains_contradiction(review):
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor']
    
    # Check if any negative word is present in the review
    has_negative = any(word in review.lower() for word in negative_words)

    return has_negative

if user_review:
    st.write('## Sentiment Prediction')
    # Preprocess the user input
    user_review_cleaned = ' '.join(
        word for word in user_review.lower().split() if word.isalpha())

    # Transform user input
    user_review_vectorized = vectorizer.transform([user_review_cleaned])

    # Model selection for prediction
    model_choice = st.selectbox('Choose Model for Prediction', list(models.keys()))
    
    if model_choice:
        model = models[model_choice]
        prediction = model.predict(user_review_vectorized)[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(user_review_vectorized).max() * 100  # Convert to percentage

        # Check for contradictions
        if contains_contradiction(user_review_cleaned):
            prediction = 'Negative'
            st.write("Contradictory sentiments detected. Classifying as **Negative**.")
        else:
            # Update sentiment based on confidence score
            if confidence is not None:
                if confidence < 41:
                    prediction = 'Negative'
                elif 41 <= confidence < 70:
                    prediction = 'Neutral'
                else:
                    prediction = 'Positive'

        st.write(f"The sentiment of the review is **{prediction}**.")
        if confidence is not None:
            st.write(f"Confidence: **{confidence:.2f}%**")

    # Visualization of sentiment distribution (for demonstration)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.write(f"### {model_choice} Confusion Matrix on Test Data")
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title(f'{model_choice} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt.gcf())

else:
    st.info("Please enter a review for prediction.")
