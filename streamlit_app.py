import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer

import streamlit as st
import pandas as pd
import joblib

nltk.download('punkt')
nltk.download('stopwords')


st.write("## Intro to NLP: Quora Question Classification using TF-IDF")

st.write("""
A web app which classifies whether a given quora question is sincere or insincere
using TF IDF - A beginner's approach to NLP.
""")

st.image("images/cover.jpeg")

st.write("""
## About
<p align="justify">
<a href="https://quora.com/" style="text-decoration: None">Quora</a> is a platform that empowers people to learn from each other. On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. <br />
This project is focused on predicting whether a given quora question is sincere or insincere. According to the data description on <a href="https://www.kaggle.com/competitions/quora-insincere-questions-classification/data" style="text-decoration: None">Kaggle</a>, insincere questions are those founded upon false premises, or that intend to make a statement rather than look for helpful answers. Other characteristics that can signify that a question is insincere include:
</p>

- Has an exaggerated tone to underscore a point about a group of people
- Suggests a discriminatory idea against a protected class of people, or seeks confirmation of a stereotype
- Makes disparaging attacks/insults against a specific person or group of people
- Based on false information, or contains absurd assumptions
- Uses sexual content (incest, bestiality, pedophilia) for shock value, and not to seek genuine answers

and so on...

<p align="justify">
My objective is to utilize NLP techniques to process and analyze the textual question to create a predictive model that can classify questions as sincere or not to some level of accuracy. <br />
Part of my objective also include familiarizing myself with fundamental NLP concepts, including tokenization, lemmatization, stop words removal, stemming, TD-IDF, and so on. In addition, I aim to gain hands-on experience in applying NLP techniques to real-world text data and understanding their impact on model performance.
</p>

""", unsafe_allow_html=True)

st.write("""
Everything you need to know regarding this project including the documentation, notebook, dataset can be found in my repository on [Github](https://github.com/Oyebamiji-Micheal/Quora-Insincere-Questions-Classification-using-TF-IDF).

**Made by Oyebamiji Micheal**
""")

st.sidebar.header("User Input")

question_input = st.sidebar.text_area("Enter the question:", "")

predict_question = st.sidebar.button("Predict Question Type")


def preprocess_question(question):
    # Tokenization
    tokens = word_tokenize(question)
    
    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalnum()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Perform stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return " ".join(tokens)


def predict_input(tweet_input):
    model_joblib = joblib.load('model.joblib')
    single_input = pd.DataFrame([{'question_text': question_input}])
    single_input = model_joblib['vectorizer'].transform(single_input.question_text)
    prediction = model_joblib['model'].predict(single_input)

    return prediction


if predict_question:
    prediction = predict_input(question_input)
    st.sidebar.write(f'Classifier = XGBoost')

    if prediction[0] == 1:
        st.sidebar.write('Predicted Status = Insincere Question')
    else:
        st.sidebar.write('Predicted Status = Sincere Question')
