# handles clean text to tokenization, stemming

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def tokenize(input_csv, output_csv):
    # whitespace tokenization
    df = pd.read_csv(input_csv)
    df["Tokens"] = df["CleanText"].str.split()

    # remove stop words
    df["Tokens"] = df["Tokens"].apply(lambda x: [word for word in x if word not in stop_words])

    # stemming 
    df["Stemmed"] = df["Tokens"].apply(lambda x: [stemmer.stem(word) for word in x])

    # might add lemmatization later and compare model accuracy
    ###

    #save new csv
    df.to_csv(output_csv, index=False)
    print(f"Tokenized data saved to {output_csv}")

    return df


if __name__ == "__main__":
    tokenize("data/reddit_messages_cleaned.csv", "data/reddit_messages_tokenized.csv")