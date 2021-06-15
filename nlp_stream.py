from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import string
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')


def run():
    st.title("Predicting ODI match Result")
    k = st.text_input('')
    lis = ['first']
    lis[0] = k
    df = pd.DataFrame(ki, columns=['review'])
    sid = SentimentIntensityAnalyzer()
    df["review"] = df["Review"].apply(lambda x: sid.polarity_scores(x))
    df = pd.concat([df.drop(['sentiments'], axis=1), df['sentiments'].apply(pd.Series)], axis=1)

def main():
    run()

if __name__ == "__main__":
  main()
