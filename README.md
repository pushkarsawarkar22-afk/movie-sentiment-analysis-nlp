# 🎬 Movie Sentiment Analysis using NLP

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![NLP](https://img.shields.io/badge/NLP-NLTK-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> An end-to-end NLP project that classifies IMDB movie reviews as **Positive** or **Negative** using Machine Learning. Includes model comparison, best model selection, and a Streamlit GUI for live predictions.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Demo](#-demo)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [NLP Pipeline](#-nlp-pipeline)
- [Models Used](#-models-used)
- [Results](#-results)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [GUI Screenshots](#-gui-screenshots)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

---

## 📖 Project Overview

**Aim:** Write Python code to perform Sentiment Analysis using NLP on Movie Reviews.

**Objective:** Build a Movie Review Sentiment Analysis system that:
- Classifies reviews as **Positive** or **Negative**
- Compares multiple ML models and selects the best one (accuracy ≥ 95%)
- Deploys a **Streamlit GUI** for real-time sentiment prediction

---

## 🎥 Demo

```
Input  → "This movie was absolutely fantastic! The acting was brilliant."
Output → ✅ Positive Sentiment

Input  → "This was a complete waste of time. Terrible plot and boring."
Output → ❌ Negative Sentiment
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Name** | IMDB Dataset of 50K Movie Reviews |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| **Size** | 50,000 reviews |
| **Classes** | Positive, Negative (balanced — 25K each) |
| **Format** | CSV (review, sentiment) |

---

## 📁 Project Structure

```
movie-sentiment-analysis-nlp/
│
├── app.py                          # Streamlit GUI application
├── NLP_Lab_08_Sentiment_Analysis.ipynb  # Google Colab notebook
├── model.pkl                       # Trained best model (pickle)
├── vectorizer.pkl                  # TF-IDF vectorizer (pickle)
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🔄 NLP Pipeline

```
Raw Review Text
      │
      ▼
1. Remove HTML Tags       (<br/> → space)
      │
      ▼
2. Lowercase              (THIS → this)
      │
      ▼
3. Remove Special Chars   (punctuation, numbers removed)
      │
      ▼
4. Negation Handling      (not good → not_good)
      │
      ▼
5. Remove Stopwords       (the, is, at... removed)
      │
      ▼
6. Lemmatization          (running → run, movies → movie)
      │
      ▼
7. TF-IDF Vectorization   (50,000 features, unigram+bigram+trigram)
      │
      ▼
8. ML Model Prediction    (Positive / Negative)
```

---

## 🤖 Models Used

| Model | Description |
|-------|-------------|
| **Naive Bayes** | MultinomialNB with alpha=0.1 |
| **Logistic Regression** | C=5, solver=saga, max_iter=1000 |
| **Random Forest** | 300 estimators |
| **SVM** | LinearSVC with C=0.5 |
| **Voting Ensemble** | LR + Calibrated SVM + NB (soft voting) |

---

## 📈 Results

| Model | Accuracy | Meets 95% Target |
|-------|----------|-----------------|
| Voting Ensemble (LR+SVM+NB) | ~95-96% | ✅ YES |
| Logistic Regression | ~93-94% | ❌ NO |
| SVM | ~93% | ❌ NO |
| Naive Bayes | ~91% | ❌ NO |
| Random Forest | ~90% | ❌ NO |

**Best Model Selected:** Voting Ensemble (LR + SVM + NB)

### TF-IDF Configuration (Key to High Accuracy)
```python
TfidfVectorizer(
    max_features=50000,   # large vocabulary
    ngram_range=(1, 3),   # unigram + bigram + trigram
    sublinear_tf=True,    # log normalization
    min_df=2,
    max_df=0.95
)
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/movie-sentiment-analysis-nlp.git
cd movie-sentiment-analysis-nlp
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK data
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

---

## ▶️ How to Run

### Option 1 — Run Streamlit GUI locally
```bash
streamlit run app.py
```
Then open → `http://localhost:8501`

### Option 2 — Run in Google Colab
1. Open `NLP_Lab_08_Sentiment_Analysis.ipynb` in Google Colab
2. Run all cells from top to bottom
3. Use localtunnel or ngrok to access the GUI

```python
# In Colab — use localtunnel
!npm install -g localtunnel -q
!streamlit run app.py &
!lt --port 8501
```

---

## 🖥️ GUI Screenshots

### Home Screen
> User enters a movie review in the text area and clicks **Analyze**

### Positive Prediction
> Green banner shows **"Positive Sentiment"** with key influencing words

### Negative Prediction  
> Red banner shows **"Negative Sentiment"** with key influencing words

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Core programming language |
| **NLTK** | Tokenization, stopwords, lemmatization |
| **Scikit-learn** | ML models, TF-IDF, evaluation |
| **Pandas / NumPy** | Data manipulation |
| **Matplotlib / Seaborn** | Visualization (confusion matrix, bar charts) |
| **Streamlit** | GUI web application |
| **Pickle** | Model serialization |
| **Google Colab** | Development environment |
| **Kaggle** | Dataset source |

---

## 📦 requirements.txt

```
pandas
numpy
nltk
scikit-learn
matplotlib
seaborn
streamlit
kagglehub
```

---

## 👨‍💻 Author

**Your Name**
- Pushkar Sawarkar
---


---

⭐ If you found this helpful, please give it a star on GitHub!
