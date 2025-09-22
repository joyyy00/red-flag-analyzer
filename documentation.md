
Setup & Data Collection
- Goal: Detect red flag vs neutral/green flag messages in relationship related texts such as Reddit, Tinder, etc.
- Data Source: Scraped 100 Reddit posts into a CSV file using PRAW (connect_reddit_api.py, reddit_messages.csv).
- Struggles: Difficult to find data directly related to relationship text messages.
- Data Labeling: Used OpenAI's API to automatically label each text as red_flag or green_flag (also meaning neutral) for effeciency (connect_openai_api.py, reddit_messages_labeled.csv).
- 
Data Pipeline
- Data Cleaning & Preprocesing: Cleaned text using a script (data_cleaning.py)
- Tokenization: Tokenized cleaned text into word lists by whitespace tokenization, removed stopwords using NLTK, applied stemming using PorterStemmer, yet to apply lemmatization (data_tokenization.py, reddit_messages_tokenized.csv)
- Vectorization/Embedding:
- 
