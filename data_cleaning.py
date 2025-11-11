# handles raw to clean text

import pandas as pd
import re
import string
import unicodedata
import contractions

def clean_text(text):
    # case standardization
    text = text.lower()

    # standardize accent characters
    text = unicodedata.normalize("NFKD", text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # remove URLs
    text = re.sub(r'https?:\S*', '', text)

    # keep letters, numbers, spaces, and ! ?  
    text = re.sub(r"[^a-z0-9!?\. ]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # normalize ellipses 
    text = re.sub(r"\.\.\.+", " ... ", text)
    
    # expanding contractions
    expanded_words = [contractions.fix(word)
                      for word in text.split()]
    text = " ".join(expanded_words) 

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
