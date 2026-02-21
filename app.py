import streamlit as st
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (first time only)
nltk.download('stopwords')
nltk.download('wordnet')

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Restaurant Sentiment Analysis", page_icon="🍽")

st.title("🍽 Restaurant Review Sentiment Analysis")
st.write("Enter a restaurant review to predict whether it is Positive or Negative.")

# ---------------------------
# Load Saved Model & Vectorizer
# ---------------------------
model = pickle.load(open("bernoulli_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------------------
# Text Cleaning Function
# ---------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)

# ---------------------------
# User Input Section
# ---------------------------
user_review = st.text_area("✍️ Write your review here:")

if st.button("Predict Sentiment"):

    if user_review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        cleaned_input = clean_text(user_review)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]

        st.subheader("Prediction Result:")

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😡 Negative Review")