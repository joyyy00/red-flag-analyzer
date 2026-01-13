# handles raw to clean text

import pandas as pd
import re
import string
import unicodedata
import contractions

def clean_text(text):
    # case standardization
    text = text.lower()

    # remove URLs
    text = re.sub(r'https?:\S*', '', text)

    # standardize accent characters
    text = unicodedata.normalize("NFKD", text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # expanding contractions
    text = contractions.fix(text)
    # expanded_words = [contractions.fix(word) for word in text.split()]
    # text = ' '.join(expanded_words) 

    # remove age/gender patterns
    text = re.sub(r'\b\d{2}[mf]\b', ' ', text)  #ex) 25m
    text = re.sub(r'\b[mf]\d{2}\b', ' ', text)  #ex) m25
    text = re.sub(r'[\(\[]\s*\d{2}[mf]\s*[\)\]]', ' ', text) #ex) ( 22m )

    # keep letters, spaces, and ! ?  
    text = re.sub(r'[^a-z!?\. ]', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # normalize ellipses 
    text = re.sub(r'\.\.\.+', ' ... ', text)

    return text


def main():
    # view labeled data
    print("Original data")
    df = pd.read_csv("reddit_messages_labeled.csv")
    print(df.head(), "\n")
    df.info()
    print("-----------------------------------------------------------")

    # remove empty rows and duplicates
    new_df = df.dropna(subset=["Text"]).drop_duplicates(subset=["Text"]).copy()

    # clean the text col and create new dataframe
    new_df["CleanText"] = new_df["Text"].apply(clean_text)

    print("Cleaned data")
    print(new_df.head(), "\n")
    new_df.info()

    #save cleaned data
    new_df.to_csv("reddit_messages_cleaned.csv", index=False)

if __name__ == "__main__":
    main()
