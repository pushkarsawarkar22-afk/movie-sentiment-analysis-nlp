import streamlit as st
import pickle
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Load NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load trained model & vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Movie Sentiment Analyzer", layout="centered")

# CSS Styling
st.markdown("""
<style>
.title    { text-align:center; font-size:34px; font-weight:600; }
.subtitle { text-align:center; color:gray; margin-bottom:25px; }
.result-pos { background:#16a34a; color:white; padding:14px;
              border-radius:8px; text-align:center; font-size:18px; margin-top:15px; }
.result-neg { background:#dc2626; color:white; padding:14px;
              border-radius:8px; text-align:center; font-size:18px; margin-top:15px; }
.explain-box { background:#f1f5f9; padding:15px; border-radius:10px; margin-top:15px; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">Movie Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">NLP-based sentiment classification system</div>', unsafe_allow_html=True)

# Text Input
review = st.text_area("Enter your movie review", height=150)

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    tokens = text.split()
    new_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "not" and i+1 < len(tokens):
            new_tokens.append("not_" + tokens[i+1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    tokens = new_tokens
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# Explanation function
def explain_prediction(text):
    feature_names = np.array(vectorizer.get_feature_names_out())
    vector = vectorizer.transform([text])
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        indices = vector.nonzero()[1]
        important = sorted(
            [(feature_names[i], coefs[i]) for i in indices],
            key=lambda x: abs(x[1]), reverse=True
        )
        important = [w for w in important if abs(w[1]) > 0.5][:5]
        return important
    return []

# Predict button
if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        cleaned = preprocess(review)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == "positive":
            st.markdown('<div class="result-pos">Positive Sentiment</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-neg">Negative Sentiment</div>', unsafe_allow_html=True)

        st.markdown("### Model Explanation")
        st.caption("Key words influencing the prediction")
        words = explain_prediction(cleaned)
        if words:
            st.markdown('<div class="explain-box">', unsafe_allow_html=True)
            for word, weight in words:
                impact = "Positive" if weight > 0 else "Negative"
                st.write(f"• **{word}** -> {impact} influence")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.write("Explanation not available for this model.")

st.markdown("---")
st.markdown("Sentiment Analysis System | NLP Lab 08")
