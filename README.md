# Red Flag Analyzer 

A machine learning text classification pipeline that scrapes relationship-related posts from Reddit, automatically labels them using the OpenAI API, and trains a custom Logistic Regression model with TF-IDF vectorization to classify messages as a "red flag" or "green flag/neutral."

## Project Structure

*   **`data/`**: Contains the raw and cleaned datasets (e.g., `reddit_messages.csv`, `reddit_messages_cleaned.csv`).
*   **`src/`**: Contains all Python scripts for scraping, labeling, cleaning, and modeling.

## Model Details
* **Algorithm:** Custom-built Binomial Logistic Regression with L2 Regularization.
* **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency).
* **Loss Function:** Binary Cross Entropy Loss (Negative Log Likelihood).
